"""Control experiment: train same LoRA on 21 MBPP synthetic-corruption pairs (same format as bootstrap).
If trained matches bootstrap (+48) → effect was format. If much smaller → bootstrap content is doing real work.
"""
import os, sys, json, time, re, gc, random, subprocess, tempfile, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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


def extract_code(text):
    if "```python" in text: text = text.split("```python", 1)[1]
    elif "```" in text: text = text.split("```", 1)[1]
    if "```" in text: text = text.split("```", 1)[0]
    return text.strip()


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


def humaneval_full(model, tok):
    he = list(load_dataset("openai_humaneval", split="test"))
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


# Synthetic corruptions
def corrupt(code, rng):
    """Apply a random corruption. Return (broken, description) or (None, None)."""
    options = []
    if "<=" in code: options.append(("lte_to_lt", code.replace("<=", "<", 1), "swapped <= to <"))
    if "==" in code: options.append(("eq_to_neq", code.replace("==", "!=", 1), "flipped == to !="))
    m = re.search(r"range\((\w+)\)", code)
    if m: options.append(("range_off", code.replace(m.group(0), f"range({m.group(1)}+1)", 1), "off-by-one in range"))
    m = re.search(r"return\s+([\w\.\[\]]+)", code, re.MULTILINE)
    if m: options.append(("ret_neg", code.replace(m.group(0), f"return -{m.group(1)}", 1), "negated return"))
    m = re.search(r"(\w+)\s*\+\s*(\w+)", code)
    if m: options.append(("plus_minus", code.replace(m.group(0), f"{m.group(1)} - {m.group(2)}", 1), "+ to -"))
    if not options: return None, None, None
    name, broken, desc = rng.choice(options)
    return broken, desc, name


def make_mbpp_pairs(n_target=21, seed=42):
    """From MBPP train, create (broken, error, fixed) corruption pairs that pass tests on canonical."""
    rng = random.Random(seed)
    mbpp_train = list(load_dataset("mbpp", "sanitized", split="train"))
    rng.shuffle(mbpp_train)

    # Reformat to look like our bootstrap pairs (signature, tests, broken, error, fixed)
    pairs = []
    for p in mbpp_train:
        sol = p["code"]
        tests = p["test_list"]
        # Canonical must pass tests
        ok_canon, _ = run_python(sol + "\n\n" + "\n".join(tests))
        if not ok_canon: continue
        # Try a corruption
        broken, desc, _ = corrupt(sol, rng)
        if broken is None or broken == sol: continue
        ok_broken, err = run_python(broken + "\n\n" + "\n".join(tests))
        if ok_broken: continue  # wasn't a real corruption
        # Build signature stub from def line + docstring
        m = re.match(r"(def\s+\w+\([^)]*\):)", sol)
        if not m: continue
        sig_line = m.group(1)
        # Pull docstring if present
        lines = sol.split("\n")
        sig_block = sig_line
        for i, l in enumerate(lines):
            if l.startswith("def "):
                # Look for docstring
                for j in range(i+1, min(i+5, len(lines))):
                    s = lines[j].strip()
                    if s.startswith('"""') and s.endswith('"""') and len(s) > 6:
                        sig_block = sig_line + "\n    " + s
                        break
                    if s.startswith('"""'):
                        # multi-line
                        doc_lines = [s]
                        for k in range(j+1, len(lines)):
                            doc_lines.append(lines[k])
                            if '"""' in lines[k]:
                                break
                        sig_block = sig_line + "\n    " + "\n    ".join(doc_lines)
                        break
                break

        pairs.append({
            "signature": sig_block, "tests": tests,
            "broken": broken, "error": err, "fixed": sol,
            "source": f"mbpp_corrupt:{desc}",
        })
        if len(pairs) >= n_target: break

    return pairs


def make_example(r, tok):
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
    ap.add_argument("--n_pairs", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="mbpp_control")
    args = ap.parse_args()

    out_dir = f"/workspace/control/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    log("generating MBPP synthetic pairs (control)")
    pairs = make_mbpp_pairs(args.n_pairs, args.seed)
    log(f"  built {len(pairs)} pairs")
    if len(pairs) < args.n_pairs:
        log(f"WARN: only {len(pairs)} pairs available")
    with open(f"{out_dir}/pairs.jsonl", "w") as fh:
        for r in pairs: fh.write(json.dumps(r) + "\n")

    log("loading Qwen/Qwen2.5-7B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda:0")

    model.eval()
    log("eval BASE on full HumanEval")
    base_corr, base_total = humaneval_full(model, tok)
    log(f"  BASE: {base_corr}/{base_total}")

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    tok.padding_side = "right"
    examples = [make_example(r, tok) for r in pairs]
    ds = HFDataset.from_list(examples)
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=10,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
        seed=args.seed,
    )
    log(f"training on {len(ds)} pairs, {args.epochs} epochs")
    Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
    log("training done")
    tok.padding_side = "left"

    model.eval()
    log("eval TRAINED on full HumanEval")
    tr_corr, tr_total = humaneval_full(model, tok)
    log(f"  TRAINED: {tr_corr}/{tr_total}")

    result = {
        "n_pairs": len(pairs), "epochs": args.epochs, "seed": args.seed,
        "data_source": "MBPP-corrupt (control)",
        "base": [base_corr, base_total], "trained": [tr_corr, tr_total],
        "delta": tr_corr - base_corr,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  CONTROL (MBPP-corrupt {len(pairs)} pairs, {args.epochs} epochs, seed {args.seed})")
    print(f"  HUMANEVAL FULL:  base={base_corr}/{base_total}  trained={tr_corr}/{tr_total}  Δ={tr_corr-base_corr:+d}")
    print(f"  time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
