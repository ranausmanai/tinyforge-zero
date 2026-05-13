"""Self-bootstrap with MBPP-train as problem seeds + vLLM on H100.

- Use MBPP train (374 problems) as PROBLEM seeds (no human solutions used).
- For each: greedy attempt. If fails, sample N attempts at temp=0.8.
- Mine at-edge pairs (broken, fixed).
- Train LoRA. Eval on HumanEval + MBPP-test.
"""
import os, json, time, re, subprocess, tempfile, argparse, gc, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def run_python(code, timeout=8):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout, text=True, cwd="/tmp")
        return r.returncode == 0, (r.stderr or "")[:200]
    except subprocess.TimeoutExpired: return False, "timeout"
    finally:
        try: os.unlink(path)
        except: pass


def vllm_gen(llm, prompts, max_new=400, temperature=0.0, n=1, stops=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                       max_tokens=max_new, n=n,
                       stop=stops or ["\nclass ", "\nif __name__", "\n\nprint", "\n\ndef "])
    out = llm.generate(prompts, sp, use_tqdm=False)
    # returns list of lists when n>1
    if n == 1:
        return [o.outputs[0].text for o in out]
    return [[c.text for c in o.outputs] for o in out]


def he_prompt(p): return p["prompt"]
def mbpp_prompt(p):
    return (f"# Task: {p['prompt']}\n"
            f"# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attempts_per", type=int, default=8)
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/selfmine_mbpp/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    log(f"loading {args.model} into vLLM")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    log(f"  loaded")

    # --- Load benchmarks
    he = list(load_dataset("openai_humaneval", split="test"))
    mbpp_test = list(load_dataset("mbpp", "sanitized", split="test"))[:200]
    mbpp_train = list(load_dataset("mbpp", "sanitized", split="train"))
    log(f"  HE: {len(he)}, MBPP-test: {len(mbpp_test)}, MBPP-train: {len(mbpp_train)}")

    # --- BASE eval
    log("=== BASE evals ===")
    t0 = time.time()
    he_outs = vllm_gen(llm, [he_prompt(p) for p in he], max_new=400)
    log(f"  HE base gen done in {time.time()-t0:.1f}s")
    base_he = 0
    for p, raw in zip(he, he_outs):
        full = p["prompt"] + raw
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        ok, _ = run_python(test_code, timeout=10)
        if ok: base_he += 1

    t1 = time.time()
    mbpp_outs = vllm_gen(llm, [mbpp_prompt(p) for p in mbpp_test], max_new=400)
    log(f"  MBPP-test base gen done in {time.time()-t1:.1f}s")
    base_mbpp = 0
    for p, raw in zip(mbpp_test, mbpp_outs):
        test_code = raw + "\n\n" + "\n".join(p["test_list"])
        ok, _ = run_python(test_code, timeout=10)
        if ok: base_mbpp += 1
    log(f"  BASE: HE={base_he}/{len(he)}  MBPP={base_mbpp}/{len(mbpp_test)}")

    # --- Mine pairs from MBPP-train
    log(f"=== mining from {len(mbpp_train)} MBPP-train problems ===")
    train_prompts = [mbpp_prompt(p) for p in mbpp_train]
    # greedy attempt
    t0 = time.time()
    greedy_outs = vllm_gen(llm, train_prompts, max_new=400)
    log(f"  greedy gen in {time.time()-t0:.1f}s")
    pairs = []
    hard_indices = []
    for i, (p, raw) in enumerate(zip(mbpp_train, greedy_outs)):
        test_code = raw + "\n\n" + "\n".join(p["test_list"])
        ok, err = run_python(test_code, timeout=8)
        if not ok:
            hard_indices.append((i, p, raw, err))
    log(f"  {len(mbpp_train) - len(hard_indices)} greedy-correct, {len(hard_indices)} hard")

    if not hard_indices:
        log("nothing to mine — base too strong"); return

    # sample N attempts per hard problem
    log(f"  sampling {args.attempts_per} attempts × {len(hard_indices)} hard problems...")
    hard_prompts = []
    for _i, p, _r, _e in hard_indices:
        hard_prompts.append(mbpp_prompt(p))
    t1 = time.time()
    sample_outs = vllm_gen(llm, hard_prompts, max_new=400, temperature=0.8, n=args.attempts_per)
    log(f"  sample gen in {time.time()-t1:.1f}s")

    t2 = time.time()
    for (idx, p, greedy_raw, err), attempts in zip(hard_indices, sample_outs):
        # check each attempt
        passes = []
        for a in attempts:
            test_code = a + "\n\n" + "\n".join(p["test_list"])
            ok, _ = run_python(test_code, timeout=8)
            if ok: passes.append(a)
        if passes:
            pairs.append({
                "problem": p["prompt"],
                "tests": p["test_list"],
                "broken": greedy_raw.strip(),
                "fixed": passes[0].strip(),
                "error": err,
            })
        if len(pairs) >= args.max_pairs: break
    log(f"  verification in {time.time()-t2:.1f}s — mined {len(pairs)} pairs")

    with open(f"{out_dir}/pairs.jsonl", "w") as fh:
        for r in pairs: fh.write(json.dumps(r) + "\n")

    if len(pairs) < 5:
        log("too few pairs — exiting"); return

    # --- Train LoRA
    log("=== TRAINING ===")
    del llm; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model

    def make_ex(r):
        user = (f"# Task: {r['problem']}\n"
                f"# Tests:\n# " + "\n# ".join(r['tests']) + "\n"
                f"# My broken attempt:\n{r['broken']}\n"
                f"# Error: {r.get('error','')[:120]}\n"
                f"# Corrected:\n")
        target = r["fixed"]
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

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    ds = HFDataset.from_list([make_ex(r) for r in pairs])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=20,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds, tokenizer=tok).train()
    log("training done")
    adapter_dir = f"{out_dir}/adapter"
    model.save_pretrained(adapter_dir)
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- TRAINED eval
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
              enable_lora=True, max_lora_rank=16)
    lora_req = LoRARequest("tf_adapter", 1, adapter_dir)
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=400, stop=["\nclass ", "\nif __name__", "\n\nprint", "\n\ndef "])

    log("=== TRAINED evals ===")
    t0 = time.time()
    he_outs = [o.outputs[0].text for o in llm.generate([he_prompt(p) for p in he], sp, lora_request=lora_req, use_tqdm=False)]
    log(f"  HE trained gen in {time.time()-t0:.1f}s")
    tr_he = 0
    for p, raw in zip(he, he_outs):
        full = p["prompt"] + raw
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        ok, _ = run_python(test_code, timeout=10)
        if ok: tr_he += 1

    t1 = time.time()
    mbpp_outs = [o.outputs[0].text for o in llm.generate([mbpp_prompt(p) for p in mbpp_test], sp, lora_request=lora_req, use_tqdm=False)]
    log(f"  MBPP-test trained gen in {time.time()-t1:.1f}s")
    tr_mbpp = 0
    for p, raw in zip(mbpp_test, mbpp_outs):
        test_code = raw + "\n\n" + "\n".join(p["test_list"])
        ok, _ = run_python(test_code, timeout=10)
        if ok: tr_mbpp += 1

    result = {
        "model": args.model, "n_pairs": len(pairs),
        "humaneval": {"base": base_he, "trained": tr_he, "delta": tr_he-base_he, "n": len(he)},
        "mbpp": {"base": base_mbpp, "trained": tr_mbpp, "delta": tr_mbpp-base_mbpp, "n": len(mbpp_test)},
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — MBPP-train SEEDED ({len(pairs)} pairs)")
    print(f"  HumanEval:  base={base_he}/{len(he)}  trained={tr_he}/{len(he)}  Δ={tr_he-base_he:+d}")
    print(f"  MBPP:       base={base_mbpp}/{len(mbpp_test)}  trained={tr_mbpp}/{len(mbpp_test)}  Δ={tr_mbpp-base_mbpp:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
