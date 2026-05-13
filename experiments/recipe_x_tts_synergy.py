"""Compound recipe + TTS: train recipe, then measure best-of-N on TOP of recipe-trained model.
Tests if recipe-trained model has BETTER sample diversity / quality at inference."""
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


def mbpp_prompt(p): return f"# Task: {p['prompt']}\n# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n"
def he_prompt(p): return p["prompt"]


def he_score_outputs(he, outs):
    c = 0
    for p, raw in zip(he, outs):
        code = raw
        if "```python" in code:
            code = code.split("```python",1)[1]
            if "```" in code: code = code.split("```",1)[0]
        full = p["prompt"] + "\n" + code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        if run_python(test_code, 10): c += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
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

    # 4 metrics:
    # A) raw greedy
    # B) raw + best-of-8
    # C) recipe greedy
    # D) recipe + best-of-8

    sp_g = SamplingParams(temperature=0, max_tokens=400, stop=["\nclass ", "\nif __name__", "\n\nprint"])
    sp_s = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=400, n=8,
                         stop=["\nclass ", "\nif __name__", "\n\nprint"])

    log("A) raw greedy")
    he_outs = [o.outputs[0].text for o in llm.generate([he_prompt(p) for p in he], sp_g, use_tqdm=False)]
    A_raw_greedy = he_score_outputs(he, he_outs)
    log(f"  raw greedy: {A_raw_greedy}/{len(he)}")

    log("B) raw best-of-8")
    he_samples = llm.generate([he_prompt(p) for p in he], sp_s, use_tqdm=False)
    B_raw_bo8 = 0
    for p, outset in zip(he, he_samples):
        for o in outset.outputs:
            code = o.text
            if "```python" in code:
                code = code.split("```python",1)[1]
                if "```" in code: code = code.split("```",1)[0]
            full = p["prompt"] + "\n" + code
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 10):
                B_raw_bo8 += 1; break
    log(f"  raw best-of-8: {B_raw_bo8}/{len(he)}")

    # Mine pairs
    log("mining pairs from MBPP-train...")
    mbpp_full = list(load_dataset("mbpp", split="train"))
    random.shuffle(mbpp_full)
    seeds = []
    for p in mbpp_full[:200]:
        prompt_text = p.get("prompt") or p.get("text", "")
        if prompt_text and p.get("test_list"):
            seeds.append({"prompt": prompt_text, "test_list": p["test_list"]})

    sp_mine = SamplingParams(temperature=0, max_tokens=400, stop=["\nclass Test", "\nif __name__"])
    g_outs = [o.outputs[0].text for o in llm.generate([mbpp_prompt(p) for p in seeds], sp_mine, use_tqdm=False)]
    hard_idx = [i for i, (p, raw) in enumerate(zip(seeds, g_outs))
                if not run_python(raw + "\n\n" + "\n".join(p["test_list"]), 8)]
    log(f"  hard: {len(hard_idx)}")
    pairs = []
    if hard_idx:
        sp_m2 = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400, n=8,
                              stop=["\nclass Test", "\nif __name__"])
        hard_prompts = [mbpp_prompt(seeds[i]) for i in hard_idx]
        sample_outs = llm.generate(hard_prompts, sp_m2, use_tqdm=False)
        for j, i in enumerate(hard_idx):
            for o in sample_outs[j].outputs:
                if run_python(o.text + "\n\n" + "\n".join(seeds[i]["test_list"]), 8):
                    pairs.append({"problem": seeds[i]["prompt"], "tests": seeds[i]["test_list"],
                                   "broken": g_outs[i].strip(), "fixed": o.text.strip()}); break
    log(f"  mined {len(pairs)} pairs")

    # Train LoRA
    del llm; gc.collect(); torch.cuda.empty_cache()
    if len(pairs) < 5:
        log("too few pairs, exit"); return

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

    # C, D
    from vllm import LLM as LLM2
    from vllm.lora.request import LoRARequest
    llm = LLM2(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
              enable_lora=True, max_lora_rank=16)
    lora_req = LoRARequest("trained", 1, adapter_dir)

    log("C) recipe greedy")
    he_outs = [o.outputs[0].text for o in llm.generate([he_prompt(p) for p in he], sp_g, lora_request=lora_req, use_tqdm=False)]
    C_rec_greedy = he_score_outputs(he, he_outs)
    log(f"  recipe greedy: {C_rec_greedy}/{len(he)}")

    log("D) recipe best-of-8")
    he_samples = llm.generate([he_prompt(p) for p in he], sp_s, lora_request=lora_req, use_tqdm=False)
    D_rec_bo8 = 0
    for p, outset in zip(he, he_samples):
        for o in outset.outputs:
            code = o.text
            if "```python" in code:
                code = code.split("```python",1)[1]
                if "```" in code: code = code.split("```",1)[0]
            full = p["prompt"] + "\n" + code
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 10):
                D_rec_bo8 += 1; break
    log(f"  recipe best-of-8: {D_rec_bo8}/{len(he)}")

    result = {
        "model": args.model, "n_pairs": len(pairs),
        "raw_greedy": A_raw_greedy, "raw_bo8": B_raw_bo8,
        "recipe_greedy": C_rec_greedy, "recipe_bo8": D_rec_bo8,
        "n": len(he), "elapsed_s": time.time() - T0,
    }
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — RECIPE × TTS COMPOUND (HumanEval, n={len(he)}, {len(pairs)} pairs)")
    print(f"  A) Raw greedy:      {A_raw_greedy:>3}/{len(he)} ({100*A_raw_greedy/len(he):.1f}%)")
    print(f"  B) Raw best-of-8:   {B_raw_bo8:>3}/{len(he)} ({100*B_raw_bo8/len(he):.1f}%)")
    print(f"  C) Recipe greedy:   {C_rec_greedy:>3}/{len(he)} ({100*C_rec_greedy/len(he):.1f}%)")
    print(f"  D) Recipe best-of-8: {D_rec_bo8:>3}/{len(he)} ({100*D_rec_bo8/len(he):.1f}%)")
    print(f"  Synergy: D - max(B,C) = {D_rec_bo8 - max(B_raw_bo8, C_rec_greedy):+d} (>0 = real synergy)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
