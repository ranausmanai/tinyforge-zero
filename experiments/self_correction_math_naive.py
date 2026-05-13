"""TinyForge-Zero self-correction for MATH-500.

Recipe:
 1. Sample real MATH-train problem (no human solutions used).
 2. Model greedy-attempt → wrong. Capture as wrong_attempt.
 3. Re-prompt model: {problem} + wrong_attempt + "Wait, let me reconsider:"
    Sample 4 completions at temp=0.8.
 4. If any completion gets correct boxed answer (verified via sympy against gold),
    MINE a triple: (problem, wrong_attempt, reflection+correct).
 5. Train LoRA on full traces — model learns to catch + fix own errors.
 6. Eval on MATH-500 (test). Model naturally produces self-correction.

Key difference from rejection-sampling: training data teaches the FIX,
not just the answer. Same broken→fixed structure that worked for code.
"""
import os, json, time, re, argparse, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model
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


def chat_messages(user_content):
    return [{"role": "system", "content": "You are a careful math problem solver. If you make a mistake, catch it and correct yourself."},
            {"role": "user", "content": user_content}]


def gen_batch(model, tok, prompts, max_new=600, temperature=0.0, batch=16, prefill_texts=None):
    """If prefill_texts provided, append each to its chat-templated prompt (forcing the model to continue from there)."""
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        pref_chunk = prefill_texts[i:i+batch] if prefill_texts else [""] * len(chunk)
        texts = []
        for p, pre in zip(chunk, pref_chunk):
            msgs = chat_messages(p)
            try:
                base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                base = p
            texts.append(base + pre)
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2000).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else 1.0, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        for j in range(out.size(0)):
            outs.append(tok.decode(out[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


def math500_eval(model, tok, n=500, batch=16):
    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[:n]
    log(f"  eval on MATH-500 ({len(ds)} problems)")
    prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in ds]
    outs = gen_batch(model, tok, prompts, max_new=800, temperature=0.0, batch=batch)
    correct = 0
    for p, raw in zip(ds, outs):
        pred = extract_boxed(raw)
        if sympy_equal(pred, p["answer"]): correct += 1
    return correct, len(ds)


def make_train_example(problem, full_solution, tok):
    """Train on the full self-correction trace."""
    user = SOLVE_PROMPT.format(problem=problem)
    msgs_pre = chat_messages(user)
    msgs_full = msgs_pre + [{"role": "assistant", "content": full_solution}]
    pre = tok.apply_chat_template(msgs_pre, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs_full, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    MAX = 1536
    full_ids = full_ids[:MAX]
    labels = list(full_ids)
    n_pre = min(len(pre_ids), len(labels))
    for i in range(n_pre): labels[i] = -100
    pad = MAX - len(full_ids)
    return {"input_ids": full_ids + [tok.pad_token_id]*pad,
            "attention_mask": [1]*len(full_ids) + [0]*pad,
            "labels": labels + [-100]*pad}


def train_on_pairs(model, tok, pairs, out_dir, lr=1e-4, epochs=2, rank=16):
    log(f"  training on {len(pairs)} traces (lr={lr}, e={epochs}, r={rank})")
    lora_cfg = LoraConfig(r=rank, lora_alpha=rank*2, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    tok.padding_side = "right"
    ds = HFDataset.from_list([make_train_example(p["problem"], p["full_solution"], tok) for p in pairs])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=lr, bf16=True, logging_steps=20,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
    tok.padding_side = "left"
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--problems_per_iter", type=int, default=48)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/math500_sc/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    log("loading MATH train split")
    train_ds = []
    for cfg in ["algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]:
        try:
            sub = list(load_dataset("EleutherAI/hendrycks_math", cfg, split="train"))
            train_ds.extend(sub)
        except Exception as e:
            log(f"  warn: failed to load {cfg}: {e}")
    log(f"  {len(train_ds)} train problems")
    random.shuffle(train_ds)

    def gold_of(p):
        return extract_boxed(p.get("solution", ""))

    model.eval()
    log("INITIAL eval on MATH-500")
    base_c, base_n = math500_eval(model, tok, n=args.n_eval)
    log(f"  MATH-500 base: {base_c}/{base_n} ({100*base_c/base_n:.1f}%)")

    pairs = []
    cursor = 0

    for it in range(1, args.iterations + 1):
        log(f"--- iter {it} ---")
        # Sample problems from MATH-train
        batch_problems = []
        while len(batch_problems) < args.problems_per_iter and cursor < len(train_ds):
            p = train_ds[cursor]; cursor += 1
            g = gold_of(p)
            if g is not None: batch_problems.append({"problem": p["problem"], "gold": g})
        if not batch_problems:
            log("  exhausted train problems"); break

        # Step 1: Greedy attempt
        prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in batch_problems]
        greedy_outs = gen_batch(model, tok, prompts, max_new=600, temperature=0.0, batch=16)
        wrong_attempts = []
        for i, (p, raw) in enumerate(zip(batch_problems, greedy_outs)):
            pred = extract_boxed(raw)
            if not sympy_equal(pred, p["gold"]):
                wrong_attempts.append({"idx": i, "problem": p["problem"], "gold": p["gold"], "wrong": raw.strip()})
        log(f"  iter {it}: {len(wrong_attempts)}/{len(batch_problems)} wrong on greedy (mining candidates)")
        if not wrong_attempts:
            continue

        # Step 2: Self-correct prompt (prefill wrong attempt + reconsider tag, sample 4)
        sc_problems = []
        prefills = []
        for w in wrong_attempts:
            for _ in range(4):
                sc_problems.append(w["problem"])
                prefills.append(w["wrong"] + RECONSIDER_TAG)
        sc_prompts = [SOLVE_PROMPT.format(problem=p) for p in sc_problems]
        sc_outs = gen_batch(model, tok, sc_prompts, max_new=600, temperature=0.8, batch=16, prefill_texts=prefills)

        mined_this_iter = 0
        for j, w in enumerate(wrong_attempts):
            attempts = sc_outs[j*4:(j+1)*4]
            preds = [extract_boxed(a) for a in attempts]
            correct_idx = [k for k, pr in enumerate(preds) if sympy_equal(pr, w["gold"])]
            if correct_idx:
                # construct full trace
                fix = attempts[correct_idx[0]].strip()
                full = w["wrong"] + RECONSIDER_TAG + fix
                pairs.append({"problem": w["problem"], "wrong_attempt": w["wrong"],
                              "correction": fix, "full_solution": full})
                mined_this_iter += 1
        log(f"  iter {it}: MINED {mined_this_iter} self-correction triples — total={len(pairs)}")

        if len(pairs) >= args.max_pairs:
            log(f"  reached max_pairs={args.max_pairs}, stopping"); break

    log(f"=== mined {len(pairs)} total self-correction triples ===")
    with open(f"{out_dir}/pairs.jsonl", "w") as fh:
        for p in pairs: fh.write(json.dumps(p) + "\n")

    if not pairs:
        log("no triples — exiting"); return

    model = train_on_pairs(model, tok, pairs, out_dir)
    log("training done")

    model.eval()
    log("FINAL eval on MATH-500")
    tr_c, tr_n = math500_eval(model, tok, n=args.n_eval)
    log(f"  MATH-500 trained: {tr_c}/{tr_n} ({100*tr_c/tr_n:.1f}%)")

    result = {
        "model": args.model, "n_pairs": len(pairs),
        "base": base_c, "trained": tr_c, "n": tr_n,
        "delta": tr_c - base_c, "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — SELF-CORRECTION recipe")
    print(f"  MATH-500: base={base_c}/{tr_n}  trained={tr_c}/{tr_n}  Δ={tr_c-base_c:+d}")
    print(f"  Triples mined: {len(pairs)}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
