"""Self-correction recipe FIXED: mix wrong→fix triples WITH right→stays-right.

Previous failure: training only on wrong→fix taught model to over-doubt itself,
causing -230 regression on Qwen3-4B-Base.

Fix:
 1. Use existing wrong→fix triples (mined yesterday).
 2. Add an equal/greater number of right→stays-right examples (greedy was correct).
 3. Train on the mixed dataset → model learns WHEN to self-correct.
 4. Eval on MATH-500.

Uses vLLM on H100 for fast generation.
"""
import os, json, time, re, argparse, gc, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset
import sympy
from sympy.parsing.latex import parse_latex

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


SOLVE_PROMPT = """Solve this competition math problem. Show your reasoning, then put the final answer in \\boxed{{...}}.

Problem: {problem}

Solution:"""


RECONSIDER_TAG = "\n\nWait, let me reconsider — I think there's an error above.\n\n"


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0: return None
    start = idx + len("\\boxed{")
    depth = 1; i = start
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    if depth != 0: return None
    return text[start:i-1].strip()


def normalize(s):
    if s is None: return None
    s = s.strip()
    s = re.sub(r"^\$|\$$", "", s).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("^{\\circ}", "")
    return s.strip()


def sympy_equal(a, b):
    if a is None or b is None: return False
    a, b = normalize(a), normalize(b)
    if a == b: return True
    try:
        ea = parse_latex(a); eb = parse_latex(b)
        if sympy.simplify(ea - eb) == 0: return True
    except Exception: pass
    try:
        fa = float(a); fb = float(b)
        if abs(fa - fb) < 1e-6: return True
    except Exception: pass
    return False


def vllm_gen(llm, prompts, max_new=600, temperature=0.0, n=1):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                       max_tokens=max_new, n=n)
    out = llm.generate(prompts, sp, use_tqdm=False)
    if n == 1: return [o.outputs[0].text for o in out]
    return [[c.text for c in o.outputs] for o in out]


def math500_eval(gen_func, label):
    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    log(f"  eval MATH-500 [{label}] ({len(ds)})")
    prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in ds]
    t0 = time.time()
    outs = gen_func(prompts, max_new=800)
    log(f"    gen done in {time.time()-t0:.1f}s")
    correct = 0
    for p, raw in zip(ds, outs):
        if sympy_equal(extract_boxed(raw), p["answer"]): correct += 1
    return correct, len(ds)


def make_train_example(problem, solution, tok):
    user = SOLVE_PROMPT.format(problem=problem)
    full = user + " " + solution
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    user_ids = tok(user + " ", add_special_tokens=False)["input_ids"]
    MAX = 1536
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
    ap.add_argument("--wrong_fix_pairs", required=True, help="Existing wrong→fix triples jsonl from prior run")
    ap.add_argument("--n_positives", type=int, default=100, help="Number of right→stays-right examples to mine")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/math500_sc_v2/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM
    from transformers import AutoTokenizer
    log(f"loading {args.model} into vLLM")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    log(f"  loaded")

    # --- BASE eval
    log("=== BASE eval ===")
    base_c, base_n = math500_eval(lambda P, max_new=800: vllm_gen(llm, P, max_new=max_new), "BASE")
    log(f"  BASE: {base_c}/{base_n} ({100*base_c/base_n:.1f}%)")

    # --- Load existing wrong→fix triples
    wrong_fix = [json.loads(l) for l in open(args.wrong_fix_pairs)]
    log(f"  loaded {len(wrong_fix)} wrong→fix triples")

    # --- Mine right→stays-right positives from MATH-train
    log(f"=== mining {args.n_positives} right→stays-right positives ===")
    train_ds = []
    for cfg in ["algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]:
        try:
            sub = list(load_dataset("EleutherAI/hendrycks_math", cfg, split="train"))
            train_ds.extend(sub)
        except Exception: pass
    random.seed(42); random.shuffle(train_ds)
    log(f"  {len(train_ds)} train problems available")

    def gold_of(p):
        return extract_boxed(p.get("solution", ""))

    positives = []
    cursor = 0
    while len(positives) < args.n_positives and cursor < len(train_ds):
        batch = []
        while len(batch) < 64 and cursor < len(train_ds):
            p = train_ds[cursor]; cursor += 1
            g = gold_of(p)
            if g is not None: batch.append({"problem": p["problem"], "gold": g})
        if not batch: break

        prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in batch]
        outs = vllm_gen(llm, prompts, max_new=600, temperature=0.0)
        for p, raw in zip(batch, outs):
            if sympy_equal(extract_boxed(raw), p["gold"]):
                # right→stays-right: model wrote a clean correct solution
                positives.append({"problem": p["problem"], "solution": raw.strip()})
                if len(positives) >= args.n_positives: break
        log(f"  positives: {len(positives)} / {args.n_positives}")

    log(f"=== final dataset: {len(wrong_fix)} wrong→fix + {len(positives)} right→stays-right = {len(wrong_fix)+len(positives)} examples ===")

    with open(f"{out_dir}/positives.jsonl", "w") as fh:
        for p in positives: fh.write(json.dumps(p) + "\n")

    # --- Build training data
    train_examples = []
    # wrong→fix as full self-correction traces
    for r in wrong_fix:
        train_examples.append({
            "problem": r["problem"],
            "solution": r["full_solution"],  # already includes wrong + RECONSIDER_TAG + correct
        })
    # right→stays-right as plain solutions (no "wait" — model commits)
    for r in positives:
        train_examples.append({
            "problem": r["problem"],
            "solution": r["solution"],
        })
    random.shuffle(train_examples)

    # --- Train LoRA
    log("=== TRAINING ===")
    del llm; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    ds_train = HFDataset.from_list([make_train_example(r["problem"], r["solution"], tok) for r in train_examples])
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
    def gen_trained(prompts, max_new=800):
        sp = SamplingParams(temperature=0, max_tokens=max_new)
        return [o.outputs[0].text for o in llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)]

    log("=== TRAINED eval ===")
    tr_c, tr_n = math500_eval(gen_trained, "TRAINED")
    log(f"  TRAINED: {tr_c}/{tr_n} ({100*tr_c/tr_n:.1f}%)")

    result = {
        "model": args.model,
        "n_wrong_fix": len(wrong_fix),
        "n_positives": len(positives),
        "n_total": len(train_examples),
        "base": base_c, "trained": tr_c, "n": tr_n,
        "delta": tr_c - base_c,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — SELF-CORRECTION V2 (mixed: {len(wrong_fix)} wrong→fix + {len(positives)} right→stays)")
    print(f"  MATH-500: base={base_c}/{tr_n} ({100*base_c/tr_n:.1f}%)  trained={tr_c}/{tr_n} ({100*tr_c/tr_n:.1f}%)  Δ={tr_c-base_c:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
