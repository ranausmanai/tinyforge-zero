"""vLLM dual eval using RAW completion format (no chat template) for base models.

Recipe for non-instruct base models — uses simple completion-style prompting
that matches how base models were pretrained.
"""
import os, json, time, re, subprocess, tempfile, argparse, gc
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

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
        return r.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally:
        try: os.unlink(path)
        except: pass


def make_he_prompt(p):
    """Raw completion: just the docstring + 'def'."""
    return p["prompt"]


def make_mbpp_prompt(p):
    """Raw completion: docstring + tests + 'def'."""
    return (f"# Task: {p['prompt']}\n"
            f"# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n")


def vllm_generate(llm, prompts, max_new=400, temperature=0.0, stops=None):
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_new, stop=stops or ["\nclass ", "\nif __name__", "\nprint(", "\n#"],
    )
    out = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in out]


def vllm_generate_lora(llm, prompts, lora_req, max_new=400, temperature=0.0, stops=None):
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_new, stop=stops or ["\nclass ", "\nif __name__", "\nprint(", "\n#"],
    )
    out = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
    return [o.outputs[0].text for o in out]


def eval_humaneval(outs_func, label):
    he = list(load_dataset("openai_humaneval", split="test"))
    log(f"  HumanEval [{label}] ({len(he)})")
    prompts = [make_he_prompt(p) for p in he]
    t0 = time.time()
    outs = outs_func(prompts, max_new=400)
    log(f"    gen done in {time.time()-t0:.1f}s")
    correct = 0
    for p, raw in zip(he, outs):
        # construct full function: prompt + raw completion
        full = p["prompt"] + raw
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        if run_python(test_code, timeout=10): correct += 1
    return correct, len(he)


def eval_mbpp(outs_func, label, n=200):
    mbpp = list(load_dataset("mbpp", "sanitized", split="test"))[:n]
    log(f"  MBPP [{label}] ({len(mbpp)})")
    prompts = [make_mbpp_prompt(p) for p in mbpp]
    t0 = time.time()
    outs = outs_func(prompts, max_new=400)
    log(f"    gen done in {time.time()-t0:.1f}s")
    correct = 0
    for p, raw in zip(mbpp, outs):
        # raw is the function code
        code = raw
        if "```" in code:
            code = extract_code("```python" + code if "```python" not in code else code)
        test_code = code + "\n\n" + "\n".join(p["test_list"])
        if run_python(test_code, timeout=10): correct += 1
    return correct, len(mbpp)


def make_train_example(r, tok):
    """Raw-completion training format."""
    sig = r.get("signature", "")
    broken = r.get("broken", "")
    fixed = r.get("fixed", "")
    tests = r.get("tests", [])
    err = r.get("error", "")
    user = (f"# Task: implement {sig}\n"
            f"# Tests:\n# " + "\n# ".join(tests) + "\n"
            f"# My broken attempt:\n{broken}\n"
            f"# Error: {err}\n"
            f"# Corrected:\n")
    target = fixed
    full = user + target
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    user_ids = tok(user, add_special_tokens=False)["input_ids"]
    MAX = 1024
    full_ids = full_ids[:MAX]
    labels = list(full_ids)
    n_user = min(len(user_ids), len(labels))
    for i in range(n_user): labels[i] = -100
    pad = MAX - len(full_ids)
    return {"input_ids": full_ids + [tok.pad_token_id]*pad,
            "attention_mask": [1]*len(full_ids) + [0]*pad,
            "labels": labels + [-100]*pad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="/workspace/saved_pairs/pairs_40.jsonl")
    ap.add_argument("--n_pairs", type=int, default=40)
    ap.add_argument("--mbpp_n", type=int, default=200)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    out_dir = f"/workspace/dual_eval_raw/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    log(f"loading {args.model} into vLLM")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    log(f"  loaded")

    log("=== BASE evals ===")
    base_he, _ = eval_humaneval(lambda P, max_new=400: vllm_generate(llm, P, max_new=max_new), "BASE")
    base_mbpp, _ = eval_mbpp(lambda P, max_new=400: vllm_generate(llm, P, max_new=max_new), "BASE", n=args.mbpp_n)
    log(f"  BASE: HumanEval={base_he}/164  MBPP={base_mbpp}/{args.mbpp_n}")

    if args.skip_train:
        result = {"model": args.model, "base_humaneval": base_he, "base_mbpp": base_mbpp, "n_he": 164, "n_mbpp": args.mbpp_n, "elapsed_s": time.time()-T0}
        with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)
        return

    # Tear down vLLM, train LoRA
    log("=== TRAINING ===")
    del llm; gc.collect(); torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model

    pairs = [json.loads(l) for l in open(args.pairs)][:args.n_pairs]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    ds = HFDataset.from_list([make_train_example(r, tok) for r in pairs])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=10,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds, tokenizer=tok).train()
    log("training done")

    adapter_dir = f"{out_dir}/adapter"
    model.save_pretrained(adapter_dir)
    del model; gc.collect(); torch.cuda.empty_cache()

    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
              enable_lora=True, max_lora_rank=16)
    lora_req = LoRARequest("tf_adapter", 1, adapter_dir)

    log("=== TRAINED evals (vLLM + LoRA) ===")
    tr_he, _ = eval_humaneval(lambda P, max_new=400: vllm_generate_lora(llm, P, lora_req, max_new=max_new), "TRAINED")
    tr_mbpp, _ = eval_mbpp(lambda P, max_new=400: vllm_generate_lora(llm, P, lora_req, max_new=max_new), "TRAINED", n=args.mbpp_n)

    result = {
        "model": args.model, "n_pairs": len(pairs),
        "humaneval": {"base": base_he, "trained": tr_he, "delta": tr_he-base_he, "n": 164},
        "mbpp": {"base": base_mbpp, "trained": tr_mbpp, "delta": tr_mbpp-base_mbpp, "n": args.mbpp_n},
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — RAW completion format")
    print(f"  HumanEval:  base={base_he}/164  trained={tr_he}/164  Δ={tr_he-base_he:+d}")
    print(f"  MBPP:       base={base_mbpp}/{args.mbpp_n}  trained={tr_mbpp}/{args.mbpp_n}  Δ={tr_mbpp-base_mbpp:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
