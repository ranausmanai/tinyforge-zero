"""Self-Bootstrapping TinyForge.

Single model. No external dataset. Just a Python interpreter.

Loop:
  for iter in 1..N:
    1. Model generates K problems (function signature + tests + canonical solution)
    2. Filter: keep only those where canonical executes & tests pass
    3. Model solves each fresh (forget canonical)
    4. Verify against tests → identify failures
    5. Model repairs each failure (one shot, with error)
    6. Verify repairs → collect (broken, fixed) pairs
    7. Periodically: LoRA-train on accumulated pairs
    8. Periodically: eval on held-out HumanEval-mini

If accuracy on HumanEval rises without ever seeing HumanEval problems → recipe works.
"""
import os, sys, json, time, re, gc, subprocess, tempfile, argparse, random, math
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_code(text):
    if "```python" in text: text = text.split("```python", 1)[1]
    elif "```" in text: text = text.split("```", 1)[1]
    if "```" in text: text = text.split("```", 1)[0]
    return text.strip()


def run_python(code, timeout=8):
    """Run code in subprocess. Return (passed, stderr_or_msg)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout, text=True, cwd="/tmp")
        if r.returncode == 0: return True, ""
        err = (r.stderr or r.stdout).strip().splitlines()
        return False, "\n".join(err[-3:])[:300]
    except subprocess.TimeoutExpired: return False, "timeout"
    finally:
        try: os.unlink(path)
        except: pass


def gen_batch(model, tok, prompts, max_new=400, temperature=0.7, batch=8):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "system", "content": "You are a Python coder."},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else 1.0, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        for j in range(out.size(0)):
            outs.append(tok.decode(out[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


PROBLEM_GEN_PROMPT = """Generate ONE simple Python coding problem with a clear function spec and 3 test assertions.

Output format (exactly one ```python block):

```python
def {function_name}({args}):
    \"\"\"{one-line description of what the function does}\"\"\"
    {implementation}

# tests
assert {function_name}(...) == ...
assert {function_name}(...) == ...
assert {function_name}(...) == ...
```

Make the function specific and concrete. The function should be 3-15 lines. Tests must verify the function works correctly. Output ONLY the code block."""


def parse_generated_problem(raw_code):
    """Split into (function_signature_with_docstring, full_solution_code, test_lines).
    Returns None if parsing fails or it's malformed."""
    code = raw_code.strip()
    if "def " not in code: return None

    # Find first def
    lines = code.split("\n")
    func_start = None
    for i, l in enumerate(lines):
        if l.startswith("def "):
            func_start = i; break
    if func_start is None: return None

    # Find tests (assert lines after the def block)
    tests = []
    in_def_body = False
    def_end = None
    for i in range(func_start, len(lines)):
        l = lines[i]
        if l.startswith("def ") and i > func_start: break
        if l.startswith("assert "):
            tests.append(l)
            if def_end is None: def_end = i
        elif tests and not l.strip().startswith(("#", "assert", "")):
            break

    if len(tests) < 2: return None
    if def_end is None: def_end = len(lines)

    full_solution = "\n".join(lines[func_start:def_end]).strip()
    if len(full_solution) < 30: return None

    # Build function signature stub for re-implementation
    # Find docstring if present
    sig_lines = []
    for i in range(func_start, def_end):
        l = lines[i]
        sig_lines.append(l)
        if i > func_start and l.strip().endswith('"""') and ('"""' in lines[i-1] or '"""' in l[:l.rfind('"""')]):
            break
        if i > func_start and l.strip().startswith('"""') and l.strip().endswith('"""') and l.strip() != '"""':
            break
        # If no docstring, stop after the def line itself
        if i == func_start and not any('"""' in lines[j] for j in range(i, min(i+5, def_end))):
            sig_lines.append("    pass")
            break
    signature = "\n".join(sig_lines)

    # Extract function name from signature
    m = re.match(r"def\s+(\w+)\s*\(", lines[func_start])
    if not m: return None
    fn_name = m.group(1)

    return {
        "fn_name": fn_name,
        "signature": signature,
        "canonical": full_solution,
        "tests": tests,
        "raw": code,
    }


# ── Loop ────────────────────────────────────────────────────────────────

def humaneval_eval(model, tok, n=30):
    """Eval on HumanEval-mini (first N problems)."""
    he = list(load_dataset("openai_humaneval", split="test"))[:n]
    prompts = [p["prompt"] + "\n# Complete the function above." for p in he]
    outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=4)
    correct = 0
    for p, raw in zip(he, outs):
        code = extract_code(raw) if "```" in raw else raw
        # Try the model's completion combined with the prompt
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        ok, _ = run_python(test_code, timeout=10)
        if ok: correct += 1
    return correct, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--problems_per_iter", type=int, default=16)
    ap.add_argument("--train_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/bootstrap/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}")
    log(f"  loaded mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    # Initial eval
    log("INITIAL eval on HumanEval-mini")
    init_correct, init_total = humaneval_eval(model, tok, n=30)
    log(f"  HumanEval-mini base: {init_correct}/{init_total}")

    # LoRA setup (will be applied for training, base kept frozen)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    log(f"  LoRA applied; trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    accumulated_pairs = []
    eval_log = [{"iter": 0, "correct": init_correct, "total": init_total}]
    iter_stats = []

    for it in range(1, args.iterations + 1):
        it_t = time.time()
        # 1. Generate K problems
        gen_prompts = [PROBLEM_GEN_PROMPT for _ in range(args.problems_per_iter)]
        raw_problems = gen_batch(model, tok, gen_prompts, max_new=400, temperature=0.9)

        # 2. Parse + verify canonical
        valid_problems = []
        for raw in raw_problems:
            code = extract_code(raw) if "```" in raw else raw
            parsed = parse_generated_problem(code)
            if parsed is None: continue
            # Verify canonical passes its own tests
            full = parsed["canonical"] + "\n\n" + "\n".join(parsed["tests"])
            ok, _ = run_python(full)
            if ok: valid_problems.append(parsed)

        if not valid_problems:
            log(f"iter {it}: 0 valid problems generated, skipping")
            iter_stats.append({"iter": it, "valid": 0, "fails": 0, "repairs": 0})
            continue

        # 3. Model solves each fresh — N=4 sampled attempts at temp=0.8 to surface natural fails
        N_ATTEMPTS = 4
        solve_prompts = [f"Implement this function so it passes the tests below.\n\n```python\n{p['signature']}\n```\n\nTests:\n{chr(10).join(p['tests'])}\n\nOutput only the function implementation in one ```python block." for p in valid_problems]
        # Generate N attempts each (4 * len(prompts) total)
        all_solve_prompts = solve_prompts * N_ATTEMPTS
        all_attempts = gen_batch(model, tok, all_solve_prompts, max_new=400, temperature=0.8)
        # Reshape: by problem, list of N attempts
        per_problem_attempts = [all_attempts[i::len(valid_problems)] for i in range(len(valid_problems))]

        # 4-5. Mine (broken, fixed) pairs from same model's diverse outputs
        failures = []
        new_pairs = 0
        for p, attempts in zip(valid_problems, per_problem_attempts):
            broken_one = None; fixed_one = None; broken_err = None
            for raw in attempts:
                code = extract_code(raw) if "```" in raw else raw
                full = code + "\n\n" + "\n".join(p["tests"])
                ok, err = run_python(full)
                if ok and fixed_one is None:
                    fixed_one = code
                elif not ok and broken_one is None:
                    broken_one = code; broken_err = err
                if broken_one and fixed_one: break
            if broken_one is None:
                continue
            if fixed_one is not None:
                # Self-mined repair pair from same-model diverse outputs
                accumulated_pairs.append({
                    "signature": p["signature"], "tests": p["tests"],
                    "broken": broken_one, "error": broken_err, "fixed": fixed_one,
                })
                new_pairs += 1
            else:
                # All attempts failed — try one more repair pass with explicit error
                failures.append({"p": p, "broken": broken_one, "error": broken_err})

        # Optional: try repair on remaining all-failed cases
        if failures:
            repair_prompts = [f"Implement: {f['p']['signature']}\n\nTests:\n{chr(10).join(f['p']['tests'])}\n\nMy attempt:\n```python\n{f['broken']}\n```\n\nError:\n{f['error']}\n\nFix and output the corrected code only." for f in failures]
            repairs = gen_batch(model, tok, repair_prompts, max_new=400, temperature=0.8)
            for f, raw in zip(failures, repairs):
                fix = extract_code(raw) if "```" in raw else raw
                full = fix + "\n\n" + "\n".join(f["p"]["tests"])
                ok, _ = run_python(full)
                if ok:
                    accumulated_pairs.append({
                        "signature": f["p"]["signature"], "tests": f["p"]["tests"],
                        "broken": f["broken"], "error": f["error"], "fixed": fix,
                    })
                    new_pairs += 1

        log(f"iter {it}: {len(valid_problems)} valid problems, {len(failures)} failures, {new_pairs} repair pairs harvested (total: {len(accumulated_pairs)})  [{time.time()-it_t:.0f}s]")
        iter_stats.append({"iter": it, "valid": len(valid_problems), "fails": len(failures), "repairs": new_pairs, "elapsed": time.time()-it_t})

        # Save incrementally (in case of crash)
        with open(f"{out_dir}/pairs.jsonl", "w") as fh:
            for r in accumulated_pairs: fh.write(json.dumps(r) + "\n")

        # 6. Periodic training
        if it % args.train_every == 0 and len(accumulated_pairs) >= 10:
            log(f"  TRAINING on {len(accumulated_pairs)} pairs")
            tok.padding_side = "right"

            def make_example(r):
                user = f"Implement: {r['signature']}\n\nTests:\n{chr(10).join(r['tests'])}\n\nMy attempt:\n```python\n{r['broken']}\n```\n\nError:\n{r['error']}\n\nFix and output the corrected code only."
                assistant = f"```python\n{r['fixed']}\n```"
                msgs_pre = [{"role": "system", "content": "You are a Python coder."},
                            {"role": "user", "content": user}]
                msgs_full = msgs_pre + [{"role": "assistant", "content": assistant}]
                pre = tok.apply_chat_template(msgs_pre, tokenize=False, add_generation_prompt=True)
                full = tok.apply_chat_template(msgs_full, tokenize=False)
                pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
                full_ids = tok(full, add_special_tokens=False)["input_ids"]
                MAX = 1024
                full_ids = full_ids[:MAX]
                labels = list(full_ids)
                n_pre = min(len(pre_ids), len(labels))
                for i in range(n_pre): labels[i] = -100
                pad = MAX - len(full_ids)
                return {"input_ids": full_ids + [tok.pad_token_id]*pad,
                        "attention_mask": [1]*len(full_ids) + [0]*pad,
                        "labels": labels + [-100]*pad}

            ds = HFDataset.from_list([make_example(r) for r in accumulated_pairs])
            targs = TrainingArguments(
                output_dir=f"{out_dir}/ckpt_iter{it}", num_train_epochs=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                learning_rate=1e-4, bf16=True, logging_steps=20,
                save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
            )
            Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
            tok.padding_side = "left"

        # 7. Periodic eval
        if it % args.eval_every == 0:
            model.eval()
            corr, tot = humaneval_eval(model, tok, n=30)
            log(f"  HumanEval-mini @ iter {it}: {corr}/{tot}")
            eval_log.append({"iter": it, "correct": corr, "total": tot})
            model.train()

    # Final eval
    model.eval()
    final_correct, final_total = humaneval_eval(model, tok, n=30)
    eval_log.append({"iter": args.iterations, "correct": final_correct, "total": final_total, "final": True})

    # Save everything
    with open(f"{out_dir}/iter_stats.jsonl", "w") as fh:
        for r in iter_stats: fh.write(json.dumps(r) + "\n")
    with open(f"{out_dir}/eval_log.json", "w") as fh:
        json.dump(eval_log, fh, indent=2)
    with open(f"{out_dir}/pairs.jsonl", "w") as fh:
        for r in accumulated_pairs: fh.write(json.dumps(r) + "\n")

    print()
    print("=" * 70)
    print(f"  MODEL: {args.model}")
    print(f"  ITERATIONS: {args.iterations}, problems/iter: {args.problems_per_iter}")
    print(f"  TOTAL repair pairs: {len(accumulated_pairs)}")
    print(f"  HUMANEVAL-MINI:  base={init_correct}/{init_total}  final={final_correct}/{final_total}  Δ={final_correct-init_correct:+d}")
    print(f"  time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
