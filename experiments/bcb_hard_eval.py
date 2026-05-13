"""Train Qwen3-8B-Base with 40-pair recipe, eval on BigCodeBench-Hard.

BigCodeBench is harder than HumanEval (real-world Python tasks, library use).
Qwen3-8B-Base likely has headroom there (~30-45% baseline). Tests if recipe
generalizes to newer model AND harder benchmark.
"""
import os, json, time, re, subprocess, tempfile, argparse
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


def verify_bcb(code, test_code):
    runner = "\n\nif __name__ == '__main__':\n    import unittest; unittest.main(argv=['x'], exit=False, verbosity=0)\n"
    body = code + "\n\n" + test_code + runner
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(body); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=20, text=True, cwd="/tmp")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        if "OK" in out and "FAILED" not in out and "Error" not in out and r.returncode == 0:
            return True
        return False
    except subprocess.TimeoutExpired:
        return False
    finally:
        try: os.unlink(path)
        except: pass


def gen_batch(model, tok, prompts, max_new=600, temperature=0.0, batch=4):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "system", "content": "You are an expert Python coder. Output one ```python block with the complete solution."},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2000).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else 1.0, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        for j in range(out.size(0)):
            outs.append(tok.decode(out[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


def eval_bcb_hard(model, tok, label, max_n=148):
    bcb = list(load_dataset("bigcode/bigcodebench-hard", split="v0.1.4"))[:max_n]
    log(f"  BCB-Hard [{label}] ({len(bcb)})")
    prompts = [p["instruct_prompt"] for p in bcb]
    outs = gen_batch(model, tok, prompts, max_new=700, batch=4)
    correct = 0
    for i, (p, raw) in enumerate(zip(bcb, outs)):
        code = extract_code(raw) if "```" in raw else raw
        if verify_bcb(code, p["test"]): correct += 1
        if (i+1) % 20 == 0: log(f"    {label} BCB {i+1}/{len(bcb)}: {correct}")
    return correct, len(bcb)


def eval_humaneval(model, tok, label):
    he = list(load_dataset("openai_humaneval", split="test"))
    log(f"  HumanEval [{label}] ({len(he)})")
    prompts = [p["prompt"] + "\n# Complete the function above." for p in he]
    outs = gen_batch(model, tok, prompts, max_new=400, batch=4)
    correct = 0
    for i, (p, raw) in enumerate(zip(he, outs)):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(test_code); path = f.name
        try:
            r = subprocess.run(["python3", path], capture_output=True, timeout=10, text=True, cwd="/tmp")
            if r.returncode == 0: correct += 1
        except subprocess.TimeoutExpired: pass
        finally:
            try: os.unlink(path)
            except: pass
        if (i+1) % 40 == 0: log(f"    {label} HE {i+1}/{len(he)}: {correct}")
    return correct, len(he)


def make_example(r, tok):
    user = (f"Implement: {r['signature']}\n\n"
            f"Tests:\n{chr(10).join(r['tests'])}\n\n"
            f"My attempt:\n```python\n{r['broken']}\n```\n\n"
            f"Error:\n{r.get('error','')}\n\n"
            f"Fix and output the corrected code only.")
    assistant = f"```python\n{r['fixed']}\n```"
    msgs_pre = [{"role": "system", "content": "You are an expert Python coder. Output one ```python block with the complete solution."},
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="/workspace/saved_pairs/pairs_40.jsonl")
    ap.add_argument("--n_pairs", type=int, default=40)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/bcb_eval/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    model.eval()
    log("=== BASE evals ===")
    base_he, _ = eval_humaneval(model, tok, "BASE")
    base_bcb, _ = eval_bcb_hard(model, tok, "BASE")
    log(f"  BASE: HumanEval={base_he}/164  BCB-Hard={base_bcb}/148")

    pairs = [json.loads(l) for l in open(args.pairs)][:args.n_pairs]
    log(f"=== TRAINING — {len(pairs)} pairs ===")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    tok.padding_side = "right"
    ds = HFDataset.from_list([make_example(r, tok) for r in pairs])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=10,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
    log("  training done")
    tok.padding_side = "left"

    model.eval()
    log("=== TRAINED evals ===")
    tr_he, _ = eval_humaneval(model, tok, "TRAINED")
    tr_bcb, _ = eval_bcb_hard(model, tok, "TRAINED")

    result = {
        "model": args.model, "method": "warmup 40 pairs",
        "humaneval": {"base": base_he, "trained": tr_he, "delta": tr_he-base_he, "n": 164},
        "bcb_hard": {"base": base_bcb, "trained": tr_bcb, "delta": tr_bcb-base_bcb, "n": 148},
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model}")
    print(f"  HumanEval:   base={base_he}/164  trained={tr_he}/164  Δ={tr_he-base_he:+d}")
    print(f"  BCB-Hard:    base={base_bcb}/148  trained={tr_bcb}/148  Δ={tr_bcb-base_bcb:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
