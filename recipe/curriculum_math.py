"""TinyForge-Zero-Math with self-difficulty curriculum.

Novel: model + interpreter only. No external problem set, no fixed difficulty.
The model's own greedy success/failure on each problem tells the curriculum
to make it harder or easier. Mine pairs only at the edge of competence.

Loop per iter:
  1. Generate K problems at current difficulty pool
  2. For each: solve greedily (temp=0). Verify against canonical answer.
     - If correct: this problem is "easy" → ask model to amplify
     - If wrong: try N=4 sampled attempts at temp=0.8
         - If at-edge (some pass, some fail): MINE a pair
         - If all fail: this problem is "too hard" → ask model to simplify
  3. Add amplified/simplified problems back into the pool for next iter
  4. Train on accumulated pairs periodically
"""
import os, sys, json, time, re, gc, argparse, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def safe_eval(expr: str):
    try:
        if not all(c in "0123456789+-*/.()% " for c in expr): return None
        return float(eval(expr, {"__builtins__": {}}, {}))
    except: return None


def extract_answer(text: str):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m: return float(m.group(1))
    m = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", text)
    if m: return float(m.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        try: return float(matches[-1])
        except: return None
    return None


def gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=8):
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


SOLVE_PROMPT = "Solve this math problem step by step. End with the answer on a new line as: #### <number>\n\nProblem: {problem}"

GEN_PROMPT_SEED = """Generate ONE math word problem with a numerical answer. Output exactly:

PROBLEM: <a clear word problem with concrete numbers>
EXPRESSION: <a single Python arithmetic expression that evaluates to the answer>
ANSWER: <the numerical answer>

Make problems grade-school level."""

AMPLIFY_PROMPT = """Take this math problem and make it HARDER by adding ONE more step (e.g., another operation, a percentage, fractions, or an extra constraint). Keep the format:

Original problem: {problem}
Original answer: {answer}

Output exactly:
PROBLEM: <the harder problem>
EXPRESSION: <Python arithmetic expression for the new answer>
ANSWER: <the new numerical answer>"""

SIMPLIFY_PROMPT = """Take this math problem and make it EASIER by removing one step or simplifying numbers. Keep the format:

Original problem: {problem}
Original answer: {answer}

Output exactly:
PROBLEM: <the easier problem>
EXPRESSION: <Python arithmetic expression for the new answer>
ANSWER: <the new numerical answer>"""


def parse_problem(text: str):
    p_m = re.search(r"PROBLEM:\s*(.+?)(?:\n|EXPRESSION:)", text, re.DOTALL)
    e_m = re.search(r"EXPRESSION:\s*(.+?)(?:\n|ANSWER:)", text, re.DOTALL)
    a_m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text)
    if not (p_m and e_m and a_m): return None
    problem = p_m.group(1).strip()
    expression = e_m.group(1).strip()
    try: claimed = float(a_m.group(1))
    except: return None
    if len(problem) < 10: return None
    actual = safe_eval(expression)
    if actual is None or abs(actual - claimed) > 0.01: return None
    return {"problem": problem, "answer": claimed}


def parse_gold(answer_field: str):
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_field)
    return float(m.group(1).replace(",", "")) if m else None


def gsm8k_eval(model, tok, n=50):
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))[:n]
    log(f"  eval on GSM8K-test ({len(ds)} problems)")
    prompts = [SOLVE_PROMPT.format(problem=p["question"]) for p in ds]
    outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=8)
    correct = 0
    for p, raw in zip(ds, outs):
        gold = parse_gold(p["answer"])
        if gold is None: continue
        pred = extract_answer(raw)
        if pred is not None and abs(pred - gold) < 0.01: correct += 1
    return correct, len(ds)


def make_train_example(problem: str, solution: str, tok):
    user = SOLVE_PROMPT.format(problem=problem)
    msgs_pre = [{"role": "system", "content": "You are a careful math tutor."},
                {"role": "user", "content": user}]
    msgs_full = msgs_pre + [{"role": "assistant", "content": solution}]
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--problems_per_iter", type=int, default=8)
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = f"/workspace/curriculum/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    model.eval()
    log("INITIAL eval on GSM8K-test")
    base_correct, base_total = gsm8k_eval(model, tok, n=args.n_eval)
    log(f"  GSM8K-test base: {base_correct}/{base_total}")

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    accumulated_pairs = []  # at-edge (problem, correct_solution)
    problem_pool = []  # current pool of problems for next iter

    for it in range(1, args.iterations + 1):
        it_t = time.time()
        # 1. Generate problems if pool is empty (seed)
        if not problem_pool or it == 1:
            gen_prompts = [GEN_PROMPT_SEED for _ in range(args.problems_per_iter)]
            raw = gen_batch(model, tok, gen_prompts, max_new=300, temperature=0.9)
            seeded = [parse_problem(r) for r in raw]
            seeded = [s for s in seeded if s]
            problem_pool.extend(seeded)
            log(f"iter {it}: seeded {len(seeded)} fresh problems (pool={len(problem_pool)})")

        # 2. Pick K problems to attempt
        random.shuffle(problem_pool)
        attempt_problems = problem_pool[:args.problems_per_iter]
        problem_pool = problem_pool[args.problems_per_iter:]  # consume

        if not attempt_problems:
            log(f"iter {it}: empty pool, regenerating"); continue

        # 3. Greedy solve to assess difficulty
        greedy_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in attempt_problems]
        greedy_outs = gen_batch(model, tok, greedy_prompts, max_new=300, temperature=0.0)
        greedy_correct = []
        for p, raw in zip(attempt_problems, greedy_outs):
            pred = extract_answer(raw)
            ok = pred is not None and abs(pred - p["answer"]) < 0.01
            greedy_correct.append(ok)

        n_easy = sum(greedy_correct)
        log(f"iter {it}: {n_easy}/{len(attempt_problems)} solved greedily")

        new_pairs = 0
        amplify_targets = []
        simplify_targets = []
        for p, easy in zip(attempt_problems, greedy_correct):
            if easy:
                # too easy → amplify next round
                amplify_targets.append(p)
            else:
                # try sampled attempts to find at-edge
                solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"])] * 4
                atts = gen_batch(model, tok, solve_prompts, max_new=300, temperature=0.8)
                ok_atts = []
                for raw in atts:
                    pred = extract_answer(raw)
                    if pred is not None and abs(pred - p["answer"]) < 0.01:
                        ok_atts.append(raw.strip())
                if ok_atts:
                    # at-edge → mine pair
                    accumulated_pairs.append({"problem": p["problem"], "solution": ok_atts[0],
                                              "answer": p["answer"]})
                    new_pairs += 1
                else:
                    # too hard → simplify
                    simplify_targets.append(p)

        log(f"iter {it}: +{new_pairs} pairs (total: {len(accumulated_pairs)}). "
            f"amplify={len(amplify_targets)}, simplify={len(simplify_targets)}")

        # 4. Generate amplified/simplified versions for next iter
        if amplify_targets:
            amp_prompts = [AMPLIFY_PROMPT.format(problem=p["problem"], answer=p["answer"]) for p in amplify_targets[:args.problems_per_iter]]
            amp_outs = gen_batch(model, tok, amp_prompts, max_new=300, temperature=0.7)
            for raw in amp_outs:
                np = parse_problem(raw)
                if np: problem_pool.append(np)
        if simplify_targets:
            sim_prompts = [SIMPLIFY_PROMPT.format(problem=p["problem"], answer=p["answer"]) for p in simplify_targets[:args.problems_per_iter // 2]]
            sim_outs = gen_batch(model, tok, sim_prompts, max_new=300, temperature=0.7)
            for raw in sim_outs:
                np = parse_problem(raw)
                if np: problem_pool.append(np)

        with open(f"{out_dir}/pairs.jsonl", "w") as fh:
            for r in accumulated_pairs: fh.write(json.dumps(r) + "\n")

        log(f"iter {it} done [{time.time()-it_t:.0f}s]; pool size now {len(problem_pool)}")

        # 5. Train every N
        if it % args.train_every == 0 and len(accumulated_pairs) >= 5:
            log(f"  TRAINING on {len(accumulated_pairs)} pairs")
            tok.padding_side = "right"
            ds = HFDataset.from_list([make_train_example(r["problem"], r["solution"], tok) for r in accumulated_pairs])
            targs = TrainingArguments(
                output_dir=f"{out_dir}/ckpt", num_train_epochs=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                learning_rate=1e-4, bf16=True, logging_steps=10,
                save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
            )
            Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
            tok.padding_side = "left"
            model.eval()
            corr, tot = gsm8k_eval(model, tok, n=args.n_eval)
            log(f"  GSM8K-test @ iter {it}: {corr}/{tot}")
            model.train()

    # Final eval
    model.eval()
    final_correct, final_total = gsm8k_eval(model, tok, n=args.n_eval)

    result = {
        "model": args.model, "iterations": args.iterations,
        "n_pairs": len(accumulated_pairs),
        "base": [base_correct, base_total],
        "trained": [final_correct, final_total],
        "delta": final_correct - base_correct,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  CURRICULUM TINYFORGE-ZERO-MATH — {args.model}")
    print(f"  Self-mined pairs: {len(accumulated_pairs)}")
    print(f"  GSM8K-test:  base={base_correct}/{base_total}  trained={final_correct}/{final_total}  Δ={final_correct-base_correct:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
