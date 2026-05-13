"""TinyForge-Zero math with MATH-train-split as problem seeds.

Recipe:
 1. Sample N problems from MATH train split (NOT test).
 2. Greedy solve each. Verify with sympy against gold answer.
 3. If greedy correct → save (problem, greedy_solution) as positive.
 4. If greedy wrong, sample 4 attempts at temp=0.8.
    Some pass → mine pair: (problem, sampled_correct_solution).
 5. Repeat until max_pairs.
 6. Train LoRA on pairs.
 7. Eval on MATH-500 (test).

Uses MATH train as problem source — model still self-generates ALL solutions.
No human solutions used.
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


def gen_batch(model, tok, prompts, max_new=600, temperature=0.0, batch=16):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "system", "content": "You are a careful math problem solver."},
                    {"role": "user", "content": p}]
            try:
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            except Exception:
                texts.append(p)
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
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
    outs = gen_batch(model, tok, prompts, max_new=600, temperature=0.0, batch=batch)
    correct = 0
    for p, raw in zip(ds, outs):
        pred = extract_boxed(raw)
        if sympy_equal(pred, p["answer"]): correct += 1
    return correct, len(ds)


def make_train_example(problem, solution, tok):
    user = SOLVE_PROMPT.format(problem=problem)
    msgs_pre = [{"role": "system", "content": "You are a careful math problem solver."},
                {"role": "user", "content": user}]
    msgs_full = msgs_pre + [{"role": "assistant", "content": solution}]
    pre = tok.apply_chat_template(msgs_pre, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs_full, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    MAX = 1280
    full_ids = full_ids[:MAX]
    labels = list(full_ids)
    n_pre = min(len(pre_ids), len(labels))
    for i in range(n_pre): labels[i] = -100
    pad = MAX - len(full_ids)
    return {"input_ids": full_ids + [tok.pad_token_id]*pad,
            "attention_mask": [1]*len(full_ids) + [0]*pad,
            "labels": labels + [-100]*pad}


def train_on_pairs(model, tok, pairs, out_dir, lr=1e-4, epochs=2, rank=16):
    log(f"  training on {len(pairs)} pairs (lr={lr}, e={epochs}, r={rank})")
    lora_cfg = LoraConfig(r=rank, lora_alpha=rank*2, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    tok.padding_side = "right"
    ds = HFDataset.from_list([make_train_example(p["problem"], p["solution"], tok) for p in pairs])
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
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--problems_per_iter", type=int, default=32)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/math500_seeded/{args.tag}"
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

    model.eval()
    log("INITIAL eval on MATH-500")
    base_c, base_n = math500_eval(model, tok, n=args.n_eval)
    log(f"  MATH-500 base: {base_c}/{base_n} ({100*base_c/base_n:.1f}%)")

    pairs = []
    cursor = 0

    def gold_of(p):
        ans = p.get("solution", "")
        b = extract_boxed(ans)
        return b

    for it in range(1, args.iterations + 1):
        log(f"--- iter {it} ---")
        batch_size = args.problems_per_iter
        # Sample with gold extractable
        batch_problems = []
        while len(batch_problems) < batch_size and cursor < len(train_ds):
            p = train_ds[cursor]; cursor += 1
            gold = gold_of(p)
            if gold is not None:
                batch_problems.append({"problem": p["problem"], "gold": gold})
        if not batch_problems:
            log("  exhausted train problems"); break

        # Greedy
        prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in batch_problems]
        greedy_outs = gen_batch(model, tok, prompts, max_new=600, temperature=0.0, batch=16)
        greedy_correct, hard_idx = 0, []
        for i, (p, raw) in enumerate(zip(batch_problems, greedy_outs)):
            pred = extract_boxed(raw)
            if sympy_equal(pred, p["gold"]):
                pairs.append({"problem": p["problem"], "solution": raw.strip(), "source": "greedy"})
                greedy_correct += 1
            else:
                hard_idx.append(i)
        log(f"  iter {it}: {greedy_correct} greedy-correct, {len(hard_idx)} hard")

        # Sampled for hard
        if hard_idx:
            hard_problems = [batch_problems[i] for i in hard_idx]
            sample_prompts = []
            for p in hard_problems:
                sample_prompts.extend([SOLVE_PROMPT.format(problem=p["problem"])] * 4)
            sample_outs = gen_batch(model, tok, sample_prompts, max_new=600, temperature=0.8, batch=16)
            sampled_correct = 0
            for i, p in enumerate(hard_problems):
                attempts = sample_outs[i*4:(i+1)*4]
                preds = [extract_boxed(a) for a in attempts]
                correct_idx = [j for j, pr in enumerate(preds) if sympy_equal(pr, p["gold"])]
                if correct_idx:
                    pairs.append({"problem": p["problem"], "solution": attempts[correct_idx[0]].strip(), "source": "sampled"})
                    sampled_correct += 1
            log(f"  iter {it}: {sampled_correct} sampled-correct (from {len(hard_idx)} hard)")

        log(f"  iter {it}: pairs total = {len(pairs)}")
        if len(pairs) >= args.max_pairs:
            log(f"  reached max_pairs={args.max_pairs}, stopping")
            break

    log(f"=== mined {len(pairs)} total pairs ===")
    with open(f"{out_dir}/pairs.jsonl", "w") as fh:
        for p in pairs: fh.write(json.dumps(p) + "\n")

    if not pairs:
        log("no pairs — exiting"); return

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
    print(f"  {args.model}")
    print(f"  MATH-500: base={base_c}/{tr_n}  trained={tr_c}/{tr_n}  Δ={tr_c-base_c:+d}")
    print(f"  Pairs mined: {len(pairs)}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
