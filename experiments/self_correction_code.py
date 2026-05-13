"""Self-correction recipe for CODE. Same pattern as math sc_v2 (which gave +5 recovery).

Pipeline:
 1. MBPP-train problems (374 sanitized + extended).
 2. Greedy attempt. If passes → save as right→stays-right positive.
 3. If fails → prompt with "Wait, let me reconsider" + sample 4 at temp=0.8.
    If any pass → mine (problem, wrong, reflection, correct) self-correction trace.
 4. Train on mixed dataset.
 5. Eval HE + MBPP.

Mix teaches model: commit to right answers, fix wrong ones.
"""
import os, json, time, re, subprocess, tempfile, argparse, gc, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


RECONSIDER_TAG = "\n\n# Wait — that doesn't look right. Let me reconsider:\n\n"


def run_python(code, timeout=8):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout, text=True, cwd="/tmp")
        return r.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally:
        try: os.unlink(path)
        except: pass


def vllm_gen(llm, prompts, max_new=400, temperature=0.0, n=1, prefill_texts=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                       max_tokens=max_new, n=n,
                       stop=["\nclass Test", "\nif __name__", "\n\nprint", "\nassert "])
    if prefill_texts is None:
        out = llm.generate(prompts, sp, use_tqdm=False)
    else:
        # Each prompt is concatenated with prefill text
        full_prompts = [p + pre for p, pre in zip(prompts, prefill_texts)]
        out = llm.generate(full_prompts, sp, use_tqdm=False)
    if n == 1: return [o.outputs[0].text for o in out]
    return [[c.text for c in o.outputs] for o in out]


def he_prompt(p): return p["prompt"]
def mbpp_prompt(p):
    return f"# Task: {p['prompt']}\n# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_mining", type=int, default=300)
    ap.add_argument("--max_self_corrections", type=int, default=80)
    ap.add_argument("--max_positives", type=int, default=80)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/code_sc/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    random.seed(42)

    from vllm import LLM
    from transformers import AutoTokenizer
    log(f"loading {args.model} into vLLM")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    log(f"  loaded")

    he = list(load_dataset("openai_humaneval", split="test"))
    mbpp_test = list(load_dataset("mbpp", "sanitized", split="test"))[:100]
    mbpp_full = list(load_dataset("mbpp", split="train"))
    random.shuffle(mbpp_full)
    seeds = []
    for p in mbpp_full[:args.n_mining]:
        prompt_text = p.get("prompt") or p.get("text", "")
        if prompt_text and p.get("test_list"):
            seeds.append({"prompt": prompt_text, "test_list": p["test_list"]})
    log(f"  HE: {len(he)}, MBPP-test: {len(mbpp_test)}, mining seeds: {len(seeds)}")

    # --- BASE eval
    log("=== BASE eval ===")
    he_outs = vllm_gen(llm, [he_prompt(p) for p in he], max_new=400)
    base_he = sum(1 for p, raw in zip(he, he_outs)
                  if run_python(p["prompt"] + raw + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})", 10))
    log(f"  HE base: {base_he}/{len(he)}")
    mbpp_outs = vllm_gen(llm, [mbpp_prompt(p) for p in mbpp_test], max_new=400)
    base_mbpp = sum(1 for p, raw in zip(mbpp_test, mbpp_outs)
                    if run_python(raw + "\n\n" + "\n".join(p["test_list"]), 10))
    log(f"  MBPP base: {base_mbpp}/{len(mbpp_test)}")

    # --- Mine: greedy on all seeds
    log(f"=== mining: greedy attempt on {len(seeds)} seeds ===")
    t0 = time.time()
    greedy_outs = vllm_gen(llm, [mbpp_prompt(p) for p in seeds], max_new=400)
    log(f"  greedy gen in {time.time()-t0:.1f}s")
    t1 = time.time()
    right = []  # greedy correct (positives)
    wrong = []  # greedy wrong (candidates for self-correction)
    for p, raw in zip(seeds, greedy_outs):
        test_code = raw + "\n\n" + "\n".join(p["test_list"])
        if run_python(test_code, timeout=8):
            right.append({"problem": p["prompt"], "tests": p["test_list"], "solution": raw.strip()})
        else:
            wrong.append({"problem": p["prompt"], "tests": p["test_list"], "wrong": raw.strip()})
    log(f"  verify: {len(right)} greedy-correct, {len(wrong)} hard")

    # --- For wrong: prefill wrong + reconsider tag, sample 4 attempts
    log(f"=== self-correction sampling on {len(wrong)} hard problems ===")
    sc_pairs = []
    if wrong:
        base_prompts = [mbpp_prompt({"prompt": w["problem"], "test_list": w["tests"]}) for w in wrong]
        prefills = [w["wrong"] + RECONSIDER_TAG for w in wrong]
        # Generate 4 attempts each via temperature
        t0 = time.time()
        sc_outs = vllm_gen(llm, base_prompts, max_new=400, temperature=0.8, n=4, prefill_texts=prefills)
        log(f"  sc gen in {time.time()-t0:.1f}s")
        t1 = time.time()
        for w, attempts in zip(wrong, sc_outs):
            for a in attempts:
                test_code = a + "\n\n" + "\n".join(w["tests"])
                if run_python(test_code, timeout=8):
                    full_trace = w["wrong"] + RECONSIDER_TAG + a.strip()
                    sc_pairs.append({"problem": w["problem"], "tests": w["tests"],
                                     "full_trace": full_trace})
                    break  # one per problem
        log(f"  sc verify in {time.time()-t1:.1f}s — {len(sc_pairs)} self-correction traces")

    # Cap and sample
    random.shuffle(right); random.shuffle(sc_pairs)
    right = right[:args.max_positives]
    sc_pairs = sc_pairs[:args.max_self_corrections]
    log(f"=== final: {len(sc_pairs)} self-correction + {len(right)} right→stays-right = {len(sc_pairs)+len(right)} examples ===")

    if len(sc_pairs) + len(right) < 10:
        log("too few examples — exiting"); return

    with open(f"{out_dir}/sc_pairs.jsonl", "w") as fh:
        for r in sc_pairs: fh.write(json.dumps(r) + "\n")
    with open(f"{out_dir}/positives.jsonl", "w") as fh:
        for r in right: fh.write(json.dumps(r) + "\n")

    # --- Train LoRA on MIXED dataset
    log("=== TRAINING ===")
    del llm; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model

    train_examples = []
    for r in sc_pairs:
        train_examples.append({"problem": r["problem"], "tests": r["tests"], "target": r["full_trace"]})
    for r in right:
        train_examples.append({"problem": r["problem"], "tests": r["tests"], "target": r["solution"]})
    random.shuffle(train_examples)

    def mk_ex(r):
        user = f"# Task: {r['problem']}\n# Tests:\n# " + "\n# ".join(r['tests']) + "\n\n"
        target = r["target"]
        full = user + target
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        user_ids = tok(user, add_special_tokens=False)["input_ids"]
        MAX = 1280
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
    ds_train = HFDataset.from_list([mk_ex(r) for r in train_examples])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=20,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds_train, tokenizer=tok).train()
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
    sp = SamplingParams(temperature=0, max_tokens=500, stop=["\nclass Test", "\nif __name__"])

    log("=== TRAINED eval ===")
    he_outs = [o.outputs[0].text for o in llm.generate([he_prompt(p) for p in he], sp, lora_request=lora_req, use_tqdm=False)]
    tr_he = sum(1 for p, raw in zip(he, he_outs)
                if run_python(p["prompt"] + raw + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})", 10))
    mbpp_outs = [o.outputs[0].text for o in llm.generate([mbpp_prompt(p) for p in mbpp_test], sp, lora_request=lora_req, use_tqdm=False)]
    tr_mbpp = sum(1 for p, raw in zip(mbpp_test, mbpp_outs)
                  if run_python(raw + "\n\n" + "\n".join(p["test_list"]), 10))

    result = {
        "model": args.model,
        "n_sc": len(sc_pairs), "n_positives": len(right), "n_total": len(train_examples),
        "humaneval": {"base": base_he, "trained": tr_he, "delta": tr_he-base_he, "n": len(he)},
        "mbpp": {"base": base_mbpp, "trained": tr_mbpp, "delta": tr_mbpp-base_mbpp, "n": len(mbpp_test)},
        "elapsed_s": time.time()-T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — CODE SELF-CORRECTION ({len(sc_pairs)} sc + {len(right)} positives)")
    print(f"  HumanEval: base={base_he}/{len(he)}  trained={tr_he}/{len(he)}  Δ={tr_he-base_he:+d}")
    print(f"  MBPP:      base={base_mbpp}/{len(mbpp_test)}  trained={tr_mbpp}/{len(mbpp_test)}  Δ={tr_mbpp-base_mbpp:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
