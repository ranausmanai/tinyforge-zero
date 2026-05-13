"""TinyForge-Zero on math word problems.

Same recipe as code bootstrap, different verifier:
  - Model generates (word_problem, python_expression_for_answer) pairs
  - Python eval gives the canonical numerical answer
  - Solver gets word problem only, must produce a number
  - Compare solver's number to canonical → broken/fixed pairs
  - Train on accumulated pairs
  - Eval on GSM8K (held-out)
"""
import os, sys, json, time, re, gc, subprocess, tempfile, argparse, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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


def safe_eval(expr: str):
    """Eval a numeric Python expression. Returns float or None."""
    try:
        # Restrict to math operations
        allowed = "0123456789+-*/.()% "
        if not all(c in allowed or c.isspace() for c in expr): return None
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def extract_answer(text: str):
    """Pull a numeric answer from model output. Looks for last number or boxed."""
    # GSM8K style: "#### 42"
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m: return float(m.group(1))
    # \boxed{42}
    m = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", text)
    if m: return float(m.group(1))
    # "answer is 42" or "= 42"
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        try: return float(matches[-1])
        except: return None
    return None


def gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=4):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "system", "content": "You are a careful math tutor."},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else 1.0, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        for j in range(out.size(0)):
            outs.append(tok.decode(out[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


PROBLEM_GEN_PROMPT = """Generate ONE math word problem with a numerical answer. Output exactly this format:

PROBLEM: <a clear word problem with concrete numbers>
EXPRESSION: <a single Python arithmetic expression that evaluates to the answer, e.g. (5*3)+12>
ANSWER: <the numerical answer>

Make the problem grade-school to middle-school level. The expression must evaluate to the answer."""


def parse_generated_problem(text: str):
    """Extract (problem, expression, answer) from model output."""
    p_m = re.search(r"PROBLEM:\s*(.+?)(?:\n|EXPRESSION:)", text, re.DOTALL)
    e_m = re.search(r"EXPRESSION:\s*(.+?)(?:\n|ANSWER:)", text, re.DOTALL)
    a_m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text)
    if not (p_m and e_m and a_m): return None
    problem = p_m.group(1).strip()
    expression = e_m.group(1).strip()
    try:
        claimed = float(a_m.group(1))
    except: return None
    if len(problem) < 10 or len(expression) < 1: return None
    # Verify: expression evaluates to claimed answer
    actual = safe_eval(expression)
    if actual is None: return None
    if abs(actual - claimed) > 0.01: return None
    return {"problem": problem, "expression": expression, "answer": claimed}


SOLVE_PROMPT_TEMPLATE = """Solve this math problem step by step. End with the answer on a new line as: #### <number>

Problem: {problem}"""


def solve_and_check(model, tok, problem_text: str, gold_answer: float, n_attempts: int = 4, temperature: float = 0.7):
    """Sample N attempts, return list of (text, predicted_num, ok)."""
    prompt = SOLVE_PROMPT_TEMPLATE.format(problem=problem_text)
    outs = gen_batch(model, tok, [prompt] * n_attempts, max_new=400, temperature=temperature)
    results = []
    for raw in outs:
        pred = extract_answer(raw)
        ok = pred is not None and abs(pred - gold_answer) < 0.01
        results.append({"text": raw, "pred": pred, "ok": ok})
    return results


def gsm8k_eval(model, tok, n=200):
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))
    ds = ds[:n]
    log(f"  eval on GSM8K ({len(ds)} problems)")
    prompts = [SOLVE_PROMPT_TEMPLATE.format(problem=p["question"]) for p in ds]
    outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=4)
    correct = 0
    for p, raw in zip(ds, outs):
        # GSM8K's answer field has format "step-by-step\n#### 42"
        gold_m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", p["answer"])
        if not gold_m: continue
        gold = float(gold_m.group(1).replace(",", ""))
        pred = extract_answer(raw)
        if pred is not None and abs(pred - gold) < 0.01: correct += 1
    return correct, len(ds)


def make_train_example(r, tok):
    user = SOLVE_PROMPT_TEMPLATE.format(problem=r["problem"]) + f"\n\nMy attempt:\n{r['broken']}\n\nThis is wrong. Solve it correctly and end with #### <number>."
    assistant = r["fixed"]
    msgs_pre = [{"role": "system", "content": "You are a careful math tutor."},
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--problems_per_iter", type=int, default=16)
    ap.add_argument("--train_every", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=8)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = f"/workspace/math/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 makes physical GPU 1 appear as cuda:0
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
    log(f"  loaded mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    # Initial eval
    model.eval()
    log("INITIAL eval on GSM8K")
    init_correct, init_total = gsm8k_eval(model, tok, n=args.n_eval)
    log(f"  GSM8K base: {init_correct}/{init_total}")

    # LoRA
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    log(f"  LoRA applied, trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    accumulated_pairs = []
    eval_log = [{"iter": 0, "correct": init_correct, "total": init_total}]
    iter_stats = []

    for it in range(1, args.iterations + 1):
        it_t = time.time()
        # 1. Generate problems
        gen_prompts = [PROBLEM_GEN_PROMPT for _ in range(args.problems_per_iter)]
        raw_problems = gen_batch(model, tok, gen_prompts, max_new=300, temperature=0.9)

        # 2. Parse & verify (Python eval of expression)
        valid = []
        for raw in raw_problems:
            parsed = parse_generated_problem(raw)
            if parsed: valid.append(parsed)

        if not valid:
            log(f"iter {it}: 0 valid problems")
            iter_stats.append({"iter": it, "valid": 0, "pairs": 0})
            continue

        # 3. Mine pairs from sampled solver outputs
        new_pairs = 0
        for p in valid:
            attempts = solve_and_check(model, tok, p["problem"], p["answer"], n_attempts=4, temperature=0.7)
            ok_atts = [a for a in attempts if a["ok"]]
            bad_atts = [a for a in attempts if not a["ok"]]
            if ok_atts and bad_atts:
                accumulated_pairs.append({
                    "problem": p["problem"],
                    "answer": p["answer"],
                    "broken": bad_atts[0]["text"],
                    "fixed": ok_atts[0]["text"],
                })
                new_pairs += 1

        log(f"iter {it}: {len(valid)} valid problems, {new_pairs} pairs harvested (total: {len(accumulated_pairs)})  [{time.time()-it_t:.0f}s]")
        iter_stats.append({"iter": it, "valid": len(valid), "pairs": new_pairs, "elapsed": time.time()-it_t})

        # Save incrementally
        with open(f"{out_dir}/pairs.jsonl", "w") as fh:
            for r in accumulated_pairs: fh.write(json.dumps(r) + "\n")

        # 4. Train every N
        if it % args.train_every == 0 and len(accumulated_pairs) >= 10:
            log(f"  TRAINING on {len(accumulated_pairs)} pairs")
            tok.padding_side = "right"
            ds = HFDataset.from_list([make_train_example(r, tok) for r in accumulated_pairs])
            targs = TrainingArguments(
                output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                learning_rate=1e-4, bf16=True, logging_steps=10,
                save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
            )
            Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
            tok.padding_side = "left"

        # 5. Eval every N
        if it % args.eval_every == 0:
            model.eval()
            corr, tot = gsm8k_eval(model, tok, n=args.n_eval)
            log(f"  GSM8K @ iter {it}: {corr}/{tot}")
            eval_log.append({"iter": it, "correct": corr, "total": tot})
            model.train()

    # Final eval
    model.eval()
    final_correct, final_total = gsm8k_eval(model, tok, n=args.n_eval)
    eval_log.append({"iter": args.iterations, "correct": final_correct, "total": final_total, "final": True})

    with open(f"{out_dir}/iter_stats.jsonl", "w") as fh:
        for r in iter_stats: fh.write(json.dumps(r) + "\n")
    with open(f"{out_dir}/eval_log.json", "w") as fh:
        json.dump(eval_log, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  TINYFORGE-ZERO ON MATH ({args.model})")
    print(f"  GSM8K-mini ({final_total}):  base={init_correct}  final={final_correct}  Δ={final_correct-init_correct:+d}")
    print(f"  Total pairs mined: {len(accumulated_pairs)}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
