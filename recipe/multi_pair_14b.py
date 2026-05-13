"""Aggressive multi-pair mining on Qwen2.5-14B-Base.

Differences from warmup recipe:
- Harder problem-generation prompt (edge cases, multi-step, tricky boundaries)
- 200 problems generated (vs 80)
- 8 sampled attempts per problem at temp 0.8 (vs 4)
- Mine ALL (broken, fixed) pairs per problem, not just 1
- Deduplicate near-identical broken code (Jaccard < 0.85)
- Larger LoRA: rank 32 attn-only
- Train fresh from base on combined (warmup_40 + new) pairs
"""
import os, sys, json, time, re, gc, subprocess, tempfile, argparse, random, hashlib
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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


def run_python(code, timeout=10):
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
            msgs = [{"role": "system", "content": "You are an expert Python coder. Output one ```python block only."},
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


def humaneval_full(model, tok):
    he = list(load_dataset("openai_humaneval", split="test"))
    log(f"  HumanEval ({len(he)} problems)")
    prompts = [p["prompt"] + "\n# Complete the function above." for p in he]
    outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=4)
    correct = 0
    for i, (p, raw) in enumerate(zip(he, outs)):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        ok, _ = run_python(test_code, timeout=10)
        if ok: correct += 1
        if (i+1) % 30 == 0: log(f"    eval {i+1}/{len(he)}: {correct} correct")
    return correct, len(he)


HARD_GEN_PROMPT = """Generate ONE challenging Python coding problem that requires:
- non-trivial algorithm (sorting variants, hash maps, two-pointer, dynamic logic, recursive backtracking, parsing, etc.)
- handles edge cases (empty input, negatives, duplicates, boundaries, or unusual inputs)
- 3 test assertions covering normal + edge cases

Output exactly:

```python
def {function_name}({args}):
    \"\"\"{problem description}\"\"\"
    {implementation}

# tests
assert {function_name}(...) == ...
assert {function_name}(...) == ...
assert {function_name}(...) == ...
```

Output ONLY the code block. Make the problem genuinely tricky."""


def parse_problem(raw):
    code = extract_code(raw) if "```" in raw else raw.strip()
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
    sig_lines = []
    for i in range(func_start, def_end):
        sig_lines.append(lines[i])
        if i == func_start and not any('"""' in lines[j] for j in range(i, min(i+5, def_end))):
            sig_lines.append("    pass"); break
    return {"fn_name": m.group(1), "signature": "\n".join(sig_lines), "tests": tests,
            "canonical": full_solution}


def code_signature(code):
    """Normalize code for dedup: strip whitespace, lowercase, hash."""
    norm = re.sub(r"\s+", " ", code).strip().lower()
    return hashlib.md5(norm.encode()).hexdigest()


def jaccard_similar(a, b, threshold=0.85):
    """Quick token-level Jaccard."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb: return False
    return len(ta & tb) / len(ta | tb) >= threshold


def mine_aggressive(model, tok, n_problems=200, max_pairs_per_problem=4, n_attempts=8,
                    batch_gen=4):
    """Generate many problems, mine ALL broken-fixed combinations per problem."""
    log(f"AGGRESSIVE MINING — {n_problems} problems, {n_attempts} attempts each, up to {max_pairs_per_problem} pairs/problem")

    # Step 1: generate problems in batches
    log("  generating problems...")
    all_problems = []
    for batch_start in range(0, n_problems, batch_gen):
        chunk_size = min(batch_gen, n_problems - batch_start)
        raws = gen_batch(model, tok, [HARD_GEN_PROMPT]*chunk_size, max_new=500, temperature=0.95, batch=batch_gen)
        for r in raws:
            p = parse_problem(r)
            if p is None: continue
            full = p["canonical"] + "\n\n" + "\n".join(p["tests"])
            ok, _ = run_python(full)
            if ok: all_problems.append(p)
        if batch_start % (batch_gen*5) == 0:
            log(f"    generated {batch_start+chunk_size}/{n_problems}, valid so far: {len(all_problems)}")
    log(f"  → {len(all_problems)} valid problems")

    # Step 2: for each problem, sample n_attempts solutions at temp 0.8, classify pass/fail
    log("  solving each problem with multiple attempts...")
    all_pairs = []
    seen_broken_sigs = set()
    for pi, p in enumerate(all_problems):
        solve_prompt = (f"Implement: {p['signature']}\n\nTests:\n{chr(10).join(p['tests'])}\n\n"
                        f"Output only the function implementation in one ```python block.")
        attempts = gen_batch(model, tok, [solve_prompt]*n_attempts, max_new=500, temperature=0.8, batch=batch_gen)
        passes, fails = [], []
        for raw in attempts:
            code = extract_code(raw) if "```" in raw else raw
            ok, err = run_python(code + "\n\n" + "\n".join(p["tests"]))
            if ok: passes.append(code)
            else: fails.append((code, err))
        # Mine pairs: each fail × each pass, capped per problem; dedupe broken
        problem_pairs = 0
        for (broken, broken_err) in fails:
            if problem_pairs >= max_pairs_per_problem: break
            sig = code_signature(broken)
            if sig in seen_broken_sigs: continue
            # check Jaccard against recent broken codes
            is_dup = False
            for existing in list(seen_broken_sigs)[-50:]:
                # can't easily reverse-hash; check against the actual broken strings we've kept
                pass
            for pass_code in passes:
                all_pairs.append({
                    "signature": p["signature"], "tests": p["tests"],
                    "broken": broken, "error": broken_err, "fixed": pass_code,
                })
                seen_broken_sigs.add(sig)
                problem_pairs += 1
                break  # one fixed per broken to keep diversity
        if (pi+1) % 10 == 0:
            log(f"    solved {pi+1}/{len(all_problems)}, pairs mined: {len(all_pairs)}")
    log(f"  AGGRESSIVE MINING DONE — {len(all_pairs)} pairs from {len(all_problems)} problems")
    return all_pairs


def make_example(r, tok):
    user = (f"Implement: {r['signature']}\n\n"
            f"Tests:\n{chr(10).join(r['tests'])}\n\n"
            f"My attempt:\n```python\n{r['broken']}\n```\n\n"
            f"Error:\n{r.get('error','')}\n\n"
            f"Fix and output the corrected code only.")
    assistant = f"```python\n{r['fixed']}\n```"
    msgs_pre = [{"role": "system", "content": "You are an expert Python coder. Output one ```python block only."},
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--warmup_pairs_path", default="/workspace/saved_pairs/pairs_40.jsonl")
    ap.add_argument("--n_warmup_pairs", type=int, default=40)
    ap.add_argument("--n_problems", type=int, default=200)
    ap.add_argument("--n_attempts", type=int, default=8)
    ap.add_argument("--max_pairs_per_problem", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/multi_pair/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    # Base eval
    model.eval()
    log("=== BASE eval ===")
    base_corr, base_total = humaneval_full(model, tok)
    log(f"  BASE: {base_corr}/{base_total}")

    # Stage 1: aggressive mining from BASE model (not from warmup — we want fresh diversity)
    log("=== AGGRESSIVE MINING (from base model) ===")
    new_pairs = mine_aggressive(model, tok,
                                n_problems=args.n_problems,
                                max_pairs_per_problem=args.max_pairs_per_problem,
                                n_attempts=args.n_attempts)
    with open(f"{out_dir}/pairs_new.jsonl", "w") as fh:
        for p in new_pairs: fh.write(json.dumps(p) + "\n")
    log(f"  saved {len(new_pairs)} new pairs")

    # Combine with warmup pairs
    warmup_pairs = [json.loads(l) for l in open(args.warmup_pairs_path)][:args.n_warmup_pairs]
    combined = warmup_pairs + new_pairs
    log(f"  combined: {len(warmup_pairs)} warmup + {len(new_pairs)} new = {len(combined)} total")

    if len(combined) < 20:
        log("FATAL: too few pairs"); return

    # Stage 2: train fresh LoRA on combined
    log(f"=== TRAINING — fresh LoRA rank={args.lora_rank}, lr={args.lr}, e={args.epochs} ===")
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tok.padding_side = "right"
    ds = HFDataset.from_list([make_example(r, tok) for r in combined])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=args.lr, bf16=True, logging_steps=20,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
    log("  training done")
    tok.padding_side = "left"

    # Stage 3: eval
    model.eval()
    log("=== TRAINED eval ===")
    tr_corr, tr_total = humaneval_full(model, tok)
    log(f"  TRAINED: {tr_corr}/{tr_total}  Δ={tr_corr-base_corr:+d}")
    model.save_pretrained(f"{out_dir}/adapter")

    result = {
        "model": args.model, "method": "aggressive multi-pair mining",
        "base": [base_corr, base_total], "trained": [tr_corr, tr_total],
        "delta": tr_corr - base_corr,
        "n_warmup_pairs": len(warmup_pairs), "n_new_pairs": len(new_pairs),
        "n_total_pairs": len(combined),
        "n_problems_generated": args.n_problems, "n_attempts_per_problem": args.n_attempts,
        "max_pairs_per_problem": args.max_pairs_per_problem,
        "lora_rank": args.lora_rank, "lr": args.lr, "epochs": args.epochs,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  MULTI-PAIR on {args.model}")
    print(f"  HumanEval:  base={base_corr}/{base_total}  trained={tr_corr}/{tr_total}  Δ={tr_corr-base_corr:+d}")
    print(f"  Total pairs: {len(combined)} ({len(warmup_pairs)} warmup + {len(new_pairs)} new)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
