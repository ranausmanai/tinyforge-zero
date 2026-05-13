"""TinyForge-Zero on CODE with self-difficulty curriculum.

Loop:
  1. Generate problem (seeded fresh or amplified/simplified from pool)
  2. Greedy solve. Verify against tests.
     - If correct → easy → amplify
     - If wrong → try 4 sampled attempts
       - If at-edge (some pass, some fail) → MINE pair
       - If all fail → too hard → simplify
  3. Train periodically. Eval on HumanEval.
"""
import os, sys, json, time, re, gc, subprocess, tempfile, argparse, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
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


def gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=4):
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


SEED_GEN_PROMPT = """Generate ONE simple Python coding problem with a clear function spec and 3 test assertions.

Output exactly:

```python
def {function_name}({args}):
    \"\"\"{description}\"\"\"
    {implementation}

# tests
assert {function_name}(...) == ...
assert {function_name}(...) == ...
assert {function_name}(...) == ...
```

Output ONLY the code block."""


AMPLIFY_PROMPT = """Take this Python coding problem and make it HARDER (add an edge case, additional constraint, or trickier logic). Keep the format with function + 3 assert tests.

Original:
```python
{original}
```

Output the harder version (function + tests) in one ```python block."""


SIMPLIFY_PROMPT = """Take this Python coding problem and make it EASIER (remove an edge case, simplify the logic). Keep the format with function + 3 assert tests.

Original:
```python
{original}
```

Output the easier version (function + tests) in one ```python block."""


def parse_problem(text):
    code = extract_code(text) if "```" in text else text.strip()
    if "def " not in code: return None
    lines = code.split("\n")
    func_start = next((i for i, l in enumerate(lines) if l.startswith("def ")), None)
    if func_start is None: return None
    tests = []
    def_end = None
    for i in range(func_start, len(lines)):
        l = lines[i]
        if l.startswith("def ") and i > func_start: break
        if l.startswith("assert "):
            tests.append(l)
            if def_end is None: def_end = i
    if len(tests) < 2: return None
    if def_end is None: def_end = len(lines)
    full_solution = "\n".join(lines[func_start:def_end]).strip()
    if len(full_solution) < 30: return None
    m = re.match(r"def\s+(\w+)\s*\(", lines[func_start])
    if not m: return None
    fn_name = m.group(1)
    sig_lines = []
    for i in range(func_start, def_end):
        sig_lines.append(lines[i])
        if i == func_start and not any('"""' in lines[j] for j in range(i, min(i+5, def_end))):
            sig_lines.append("    pass"); break
        if i > func_start and '"""' in lines[i] and (i > func_start+1 and '"""' in lines[i-1] or lines[i].count('"""') >= 2):
            break
    return {"fn_name": fn_name, "signature": "\n".join(sig_lines), "tests": tests,
            "canonical": full_solution, "raw": code}


def humaneval_full(model, tok, n=164):
    he = list(load_dataset("openai_humaneval", split="test"))[:n]
    log(f"  HumanEval ({len(he)} problems)")
    prompts = [p["prompt"] + "\n# Complete the function above." for p in he]
    outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=4)
    correct = 0
    for p, raw in zip(he, outs):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        ok, _ = run_python(test_code, timeout=10)
        if ok: correct += 1
    return correct, len(he)


def make_train_example(r, tok):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--problems_per_iter", type=int, default=8)
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = f"/workspace/curriculum_code/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    model.eval()
    log("INITIAL eval on HumanEval")
    base_correct, base_total = humaneval_full(model, tok)
    log(f"  base: {base_correct}/{base_total}")

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    accumulated = []
    problem_pool = []

    for it in range(1, args.iterations + 1):
        it_t = time.time()

        if not problem_pool:
            gen_prompts = [SEED_GEN_PROMPT for _ in range(args.problems_per_iter)]
            raw = gen_batch(model, tok, gen_prompts, max_new=400, temperature=0.9)
            seeded = []
            for r in raw:
                parsed = parse_problem(r)
                if not parsed: continue
                full = parsed["canonical"] + "\n\n" + "\n".join(parsed["tests"])
                ok, _ = run_python(full)
                if ok: seeded.append(parsed)
            problem_pool.extend(seeded)
            log(f"iter {it}: seeded {len(seeded)} fresh (pool={len(problem_pool)})")

        random.shuffle(problem_pool)
        attempt_problems = problem_pool[:args.problems_per_iter]
        problem_pool = problem_pool[args.problems_per_iter:]

        if not attempt_problems:
            log(f"iter {it}: empty pool"); continue

        # Greedy solve
        greedy_prompts = [f"Implement: {p['signature']}\n\nTests:\n{chr(10).join(p['tests'])}\n\nOutput only the function in one ```python block." for p in attempt_problems]
        greedy_outs = gen_batch(model, tok, greedy_prompts, max_new=300, temperature=0.0)
        new_pairs = 0
        amp_targets = []; sim_targets = []
        for p, raw in zip(attempt_problems, greedy_outs):
            code = extract_code(raw) if "```" in raw else raw
            ok, _ = run_python(code + "\n\n" + "\n".join(p["tests"]))
            if ok:
                amp_targets.append(p)
            else:
                # at-edge check via sampling
                solve_prompt = f"Implement: {p['signature']}\n\nTests:\n{chr(10).join(p['tests'])}\n\nOutput only the function in one ```python block."
                atts = gen_batch(model, tok, [solve_prompt]*4, max_new=300, temperature=0.7)
                broken = None; broken_err = None; fixed = None
                for ra in atts:
                    c = extract_code(ra) if "```" in ra else ra
                    ok2, err = run_python(c + "\n\n" + "\n".join(p["tests"]))
                    if ok2 and fixed is None: fixed = c
                    elif not ok2 and broken is None: broken = c; broken_err = err
                    if broken and fixed: break
                if broken and fixed:
                    accumulated.append({"signature": p["signature"], "tests": p["tests"],
                                        "broken": broken, "error": broken_err, "fixed": fixed})
                    new_pairs += 1
                else:
                    sim_targets.append(p)

        log(f"iter {it}: {len(attempt_problems)} attempted, +{new_pairs} pairs (total: {len(accumulated)}). amp={len(amp_targets)}, sim={len(sim_targets)}  [{time.time()-it_t:.0f}s]")

        # Generate amplified / simplified for next iter
        if amp_targets:
            amp_prompts = [AMPLIFY_PROMPT.format(original=p["raw"]) for p in amp_targets[:args.problems_per_iter]]
            amp_outs = gen_batch(model, tok, amp_prompts, max_new=400, temperature=0.7)
            for r in amp_outs:
                parsed = parse_problem(r)
                if not parsed: continue
                full = parsed["canonical"] + "\n\n" + "\n".join(parsed["tests"])
                ok, _ = run_python(full)
                if ok: problem_pool.append(parsed)
        if sim_targets:
            sim_prompts = [SIMPLIFY_PROMPT.format(original=p["raw"]) for p in sim_targets[:args.problems_per_iter//2]]
            sim_outs = gen_batch(model, tok, sim_prompts, max_new=400, temperature=0.7)
            for r in sim_outs:
                parsed = parse_problem(r)
                if not parsed: continue
                full = parsed["canonical"] + "\n\n" + "\n".join(parsed["tests"])
                ok, _ = run_python(full)
                if ok: problem_pool.append(parsed)

        with open(f"{out_dir}/pairs.jsonl", "w") as fh:
            for r in accumulated: fh.write(json.dumps(r) + "\n")

        if it % args.train_every == 0 and len(accumulated) >= 10:
            log(f"  TRAINING on {len(accumulated)} pairs")
            tok.padding_side = "right"
            ds = HFDataset.from_list([make_train_example(r, tok) for r in accumulated])
            targs = TrainingArguments(
                output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                learning_rate=1e-4, bf16=True, logging_steps=10,
                save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
            )
            Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
            tok.padding_side = "left"
            model.eval()
            corr, tot = humaneval_full(model, tok)
            log(f"  HumanEval @ iter {it}: {corr}/{tot}  Δ={corr-base_correct:+d}")
            model.train()

    model.eval()
    final_correct, final_total = humaneval_full(model, tok)

    result = {
        "model": args.model, "iterations": args.iterations,
        "n_pairs": len(accumulated),
        "base": [base_correct, base_total],
        "trained": [final_correct, final_total],
        "delta": final_correct - base_correct,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  CURRICULUM TINYFORGE-ZERO-CODE — {args.model}")
    print(f"  HumanEval:  base={base_correct}/{base_total}  trained={final_correct}/{final_total}  Δ={final_correct-base_correct:+d}")
    print(f"  Self-mined pairs: {len(accumulated)}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
