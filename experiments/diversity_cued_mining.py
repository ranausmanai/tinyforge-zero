"""Diversity-aware mining: prompt model with multiple cognitive lenses, mine pairs WITHOUT including failed code.
Train on (problem, best_approach_summary, working_code) — minimal traces."""
import os, json, time, re, subprocess, tempfile, argparse, gc, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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


LENS_PROMPTS = [
    ("brute force iteration", "# Loop and check each case."),
    ("math formula", "# Use a closed-form formula."),
    ("hash map/set", "# Use a hashmap/set for O(1) lookup."),
    ("recursion", "# Solve recursively."),
]


def mbpp_prompt(p): return f"# Task: {p['prompt']}\n# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n"
def he_prompt(p): return p["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_mining", type=int, default=150)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(42)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    log("loaded")

    he = list(load_dataset("openai_humaneval", split="test"))
    mbpp_test = list(load_dataset("mbpp", "sanitized", split="test"))[:100]
    mbpp_full = list(load_dataset("mbpp", split="train"))
    random.shuffle(mbpp_full)
    seeds = []
    for p in mbpp_full[:args.n_mining]:
        prompt_text = p.get("prompt") or p.get("text", "")
        if prompt_text and p.get("test_list"):
            seeds.append({"prompt": prompt_text, "test_list": p["test_list"]})
    log(f"  HE: {len(he)}, MBPP-test: {len(mbpp_test)}, mining: {len(seeds)}")

    # Base eval
    sp_g = SamplingParams(temperature=0, max_tokens=400, stop=["\nclass ", "\nif __name__", "\n\nprint"])
    he_outs = [o.outputs[0].text for o in llm.generate([he_prompt(p) for p in he], sp_g, use_tqdm=False)]
    base_he = sum(1 for p, raw in zip(he, he_outs)
                  if run_python(p["prompt"] + "\n" + raw + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})", 10))
    mbpp_outs = [o.outputs[0].text for o in llm.generate([mbpp_prompt(p) for p in mbpp_test], sp_g, use_tqdm=False)]
    base_mbpp = sum(1 for p, raw in zip(mbpp_test, mbpp_outs)
                    if run_python(raw + "\n\n" + "\n".join(p["test_list"]), 10))
    log(f"BASE: HE={base_he}/{len(he)}  MBPP={base_mbpp}/{len(mbpp_test)}")

    # Mine: for each problem, generate 4 lens-cued attempts, keep one that works
    log("mining with cued diversity...")
    pairs = []
    for lens_name, lens_hint in LENS_PROMPTS:
        log(f"  lens: {lens_name}")
        # Prefill prompts with lens hint
        prefilled = []
        for s in seeds:
            base = mbpp_prompt(s) + f"# Approach: {lens_name}.\n{lens_hint}\ndef solution"
            prefilled.append(base)
        sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=300,
                           stop=["\nclass Test", "\nif __name__", "\n\nprint", "\n# Task"])
        outs = [o.outputs[0].text for o in llm.generate(prefilled, sp, use_tqdm=False)]
        # Verify each
        for s, raw in zip(seeds, outs):
            code = "def solution" + raw
            if run_python(code + "\n\n" + "\n".join(s["test_list"]), 8):
                # Greedy attempt to use as broken
                greedy = [o.outputs[0].text for o in llm.generate([mbpp_prompt(s)], sp_g, use_tqdm=False)][0]
                if not run_python(greedy + "\n\n" + "\n".join(s["test_list"]), 8):
                    pairs.append({"problem": s["prompt"], "tests": s["test_list"],
                                   "broken": greedy.strip(), "fixed": code.strip(),
                                   "lens": lens_name})
    log(f"mined {len(pairs)} pairs across lenses")

    with open(f"{args.out_dir}/pairs.jsonl", "w") as fh:
        for r in pairs: fh.write(json.dumps(r) + "\n")

    if len(pairs) < 5:
        result = {"model": args.model, "n_pairs": len(pairs), "base_he": base_he, "base_mbpp": base_mbpp}
        with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)
        return

    # Train flat
    del llm; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model

    def mk_ex(r):
        user = (f"# Task: {r['problem']}\n# Tests:\n# " + "\n# ".join(r['tests']) + "\n"
                f"# My broken attempt:\n{r['broken']}\n# Corrected:\n")
        full = user + r["fixed"]
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        user_ids = tok(user, add_special_tokens=False)["input_ids"]
        MAX = 1024
        full_ids = full_ids[:MAX]
        labels = list(full_ids); n_user = min(len(user_ids), len(labels))
        for i in range(n_user): labels[i] = -100
        pad = MAX - len(full_ids)
        return {"input_ids": full_ids + [tok.pad_token_id]*pad,
                "attention_mask": [1]*len(full_ids) + [0]*pad,
                "labels": labels + [-100]*pad}

    log("training...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    ds_train = HFDataset.from_list([mk_ex(r) for r in pairs])
    targs = TrainingArguments(
        output_dir=f"{args.out_dir}/ckpt", num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=20,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds_train, tokenizer=tok).train()
    adapter_dir = f"{args.out_dir}/adapter"
    model.save_pretrained(adapter_dir)
    del model; gc.collect(); torch.cuda.empty_cache()

    # Trained eval
    from vllm import LLM as LLM2
    from vllm.lora.request import LoRARequest
    llm = LLM2(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
              enable_lora=True, max_lora_rank=16)
    lora_req = LoRARequest("trained", 1, adapter_dir)
    he_outs = [o.outputs[0].text for o in llm.generate([he_prompt(p) for p in he], sp_g, lora_request=lora_req, use_tqdm=False)]
    tr_he = sum(1 for p, raw in zip(he, he_outs)
                if run_python(p["prompt"] + "\n" + raw + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})", 10))
    mbpp_outs = [o.outputs[0].text for o in llm.generate([mbpp_prompt(p) for p in mbpp_test], sp_g, lora_request=lora_req, use_tqdm=False)]
    tr_mbpp = sum(1 for p, raw in zip(mbpp_test, mbpp_outs)
                  if run_python(raw + "\n\n" + "\n".join(p["test_list"]), 10))

    result = {
        "model": args.model, "n_pairs": len(pairs),
        "humaneval": {"base": base_he, "trained": tr_he, "delta": tr_he-base_he, "n": len(he)},
        "mbpp": {"base": base_mbpp, "trained": tr_mbpp, "delta": tr_mbpp-base_mbpp, "n": len(mbpp_test)},
        "elapsed_s": time.time() - T0,
    }
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — DIVERSITY-CUED MINING ({len(pairs)} pairs)")
    print(f"  HE:   base={base_he}/{len(he)}  trained={tr_he}/{len(he)}  Δ={tr_he-base_he:+d}")
    print(f"  MBPP: base={base_mbpp}/{len(mbpp_test)}  trained={tr_mbpp}/{len(mbpp_test)}  Δ={tr_mbpp-base_mbpp:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
