"""Cross-domain transfer: train recipe on CODE, eval on MATH (no math training).
Tests if self-bootstrap teaches generic reasoning vs domain-specific patterns."""
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


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0: return None
    start = idx + len("\\boxed{"); depth = 1; i = start
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    if depth != 0: return None
    return text[start:i-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_domain", choices=["code", "math"], default="code")
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

    # Eval sets
    he = list(load_dataset("openai_humaneval", split="test"))[:80]
    math500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[:100]

    # Build prompts
    he_prompts = [p["prompt"] for p in he]
    math_prompts = []
    for p in math500:
        try:
            msgs = [{"role": "system", "content": "Math solver. End with \\boxed{answer}."},
                    {"role": "user", "content": f"Solve. Problem: {p['problem']}\n\nSolution:"}]
            math_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            math_prompts.append(f"Solve. Problem: {p['problem']}\n\nSolution:")

    import sympy
    from sympy.parsing.latex import parse_latex
    def sympy_eq(a, b):
        if a is None or b is None: return False
        if a.strip() == b.strip(): return True
        try:
            if sympy.simplify(parse_latex(a) - parse_latex(b)) == 0: return True
        except Exception: pass
        try:
            if abs(float(a) - float(b)) < 1e-6: return True
        except Exception: pass
        return False

    def eval_he(llm, lora_req=None):
        sp = SamplingParams(temperature=0, max_tokens=400, stop=["\nclass ", "\nif __name__", "\n\nprint"])
        outs = llm.generate(he_prompts, sp, lora_request=lora_req, use_tqdm=False) if lora_req else \
               llm.generate(he_prompts, sp, use_tqdm=False)
        outs = [o.outputs[0].text for o in outs]
        c = 0
        for p, raw in zip(he, outs):
            full = p["prompt"] + "\n" + raw
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 10): c += 1
        return c, len(he)

    def eval_math(llm, lora_req=None):
        sp = SamplingParams(temperature=0, max_tokens=800)
        outs = llm.generate(math_prompts, sp, lora_request=lora_req, use_tqdm=False) if lora_req else \
               llm.generate(math_prompts, sp, use_tqdm=False)
        outs = [o.outputs[0].text for o in outs]
        c = 0
        for p, raw in zip(math500, outs):
            if sympy_eq(extract_boxed(raw), p["answer"]): c += 1
        return c, len(math500)

    log("=== BASE evals ===")
    base_he = eval_he(llm)
    base_math = eval_math(llm)
    log(f"  base HE: {base_he[0]}/{base_he[1]}  MATH: {base_math[0]}/{base_math[1]}")

    # Mine code pairs
    log("mining code pairs...")
    mbpp_full = list(load_dataset("mbpp", split="train"))
    random.shuffle(mbpp_full)
    seeds = []
    for p in mbpp_full[:200]:
        prompt_text = p.get("prompt") or p.get("text", "")
        if prompt_text and p.get("test_list"):
            seeds.append({"prompt": prompt_text, "test_list": p["test_list"]})

    def mbpp_prompt(p): return f"# Task: {p['prompt']}\n# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n"

    sp = SamplingParams(temperature=0, max_tokens=400, stop=["\nclass Test", "\nif __name__"])
    g_outs = [o.outputs[0].text for o in llm.generate([mbpp_prompt(p) for p in seeds], sp, use_tqdm=False)]
    hard_idx = []
    for i, (p, raw) in enumerate(zip(seeds, g_outs)):
        if not run_python(raw + "\n\n" + "\n".join(p["test_list"]), 8):
            hard_idx.append(i)
    log(f"  greedy: {len(seeds)-len(hard_idx)} pass, {len(hard_idx)} hard")
    pairs = []
    if hard_idx:
        sp2 = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400, n=8,
                            stop=["\nclass Test", "\nif __name__"])
        hard_prompts = [mbpp_prompt(seeds[i]) for i in hard_idx]
        sample_outs = llm.generate(hard_prompts, sp2, use_tqdm=False)
        for j, i in enumerate(hard_idx):
            attempts = [o.text for o in sample_outs[j].outputs]
            for a in attempts:
                if run_python(a + "\n\n" + "\n".join(seeds[i]["test_list"]), 8):
                    pairs.append({"problem": seeds[i]["prompt"], "tests": seeds[i]["test_list"],
                                   "broken": g_outs[i].strip(), "fixed": a.strip()})
                    break
    log(f"  mined {len(pairs)} code pairs")

    if len(pairs) < 5:
        log("too few pairs, skipping train")
        result = {"model": args.model, "n_pairs": len(pairs),
                  "base_he": base_he[0], "base_math": base_math[0]}
        with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)
        return

    # Tear down vLLM, train LoRA
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

    log("training LoRA on code pairs...")
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
    log("training done")

    # Re-eval with adapter
    log("=== TRAINED evals ===")
    from vllm import LLM as LLM2
    from vllm.lora.request import LoRARequest
    llm = LLM2(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
              enable_lora=True, max_lora_rank=16)
    lora_req = LoRARequest("trained", 1, adapter_dir)
    tr_he = eval_he(llm, lora_req)
    tr_math = eval_math(llm, lora_req)
    log(f"  trained HE: {tr_he[0]}/{tr_he[1]}  MATH: {tr_math[0]}/{tr_math[1]}")

    result = {
        "model": args.model, "train_domain": args.train_domain,
        "n_pairs": len(pairs),
        "humaneval": {"base": base_he[0], "trained": tr_he[0], "delta": tr_he[0]-base_he[0], "n": base_he[1]},
        "math500": {"base": base_math[0], "trained": tr_math[0], "delta": tr_math[0]-base_math[0], "n": base_math[1]},
        "elapsed_s": time.time() - T0,
    }
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — CROSS-DOMAIN ({args.train_domain} train, eval HE+MATH)")
    print(f"  HE:    base={base_he[0]}/{base_he[1]}  trained={tr_he[0]}/{tr_he[1]}  Δ={tr_he[0]-base_he[0]:+d}")
    print(f"  MATH:  base={base_math[0]}/{base_math[1]}  trained={tr_math[0]}/{tr_math[1]}  Δ={tr_math[0]-base_math[0]:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
