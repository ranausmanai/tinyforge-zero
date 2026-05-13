"""Self-consistency selection: majority vote on N samples WITHOUT oracle access.
Tests if model's self-agreement is a good selector (deployable TTS without test cases)."""
import os, json, time, re, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset
from collections import Counter

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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


def normalize(s):
    if s is None: return None
    s = s.strip().lower()
    s = re.sub(r"[,$\s]", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048)
    log("loaded")

    math500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[:200]
    prompts = []
    for p in math500:
        try:
            msgs = [{"role": "system", "content": "Math solver. End with \\boxed{answer}."},
                    {"role": "user", "content": f"Solve. Problem: {p['problem']}\n\nSolution:"}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(f"Solve. Problem: {p['problem']}\n\nSolution:")

    log(f"generating {args.n_samples} samples per problem...")
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=800, n=args.n_samples)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    log(f"  gen in {time.time()-t0:.1f}s")

    import sympy
    from sympy.parsing.latex import parse_latex
    def sympy_eq(a, b):
        if a is None or b is None: return False
        if a == b: return True
        try:
            if sympy.simplify(parse_latex(a) - parse_latex(b)) == 0: return True
        except Exception: pass
        try:
            if abs(float(a) - float(b)) < 1e-6: return True
        except Exception: pass
        return False

    # Three metrics:
    # 1. Greedy: take first sample
    # 2. Oracle pass@N: any correct
    # 3. Self-consistency: majority vote on extracted boxed answer (normalize numbers/text)
    greedy_correct = 0
    oracle_correct = 0
    sc_correct = 0

    for p, outset in zip(math500, outs):
        attempts = [o.text for o in outset.outputs]
        preds = [extract_boxed(a) for a in attempts]
        # Greedy: first sample
        if sympy_eq(preds[0], p["answer"]): greedy_correct += 1
        # Oracle: any pass
        if any(sympy_eq(pr, p["answer"]) for pr in preds): oracle_correct += 1
        # Self-consistency: majority vote on normalized answer
        normalized = [normalize(pr) for pr in preds if pr is not None]
        if normalized:
            most_common, _ = Counter(normalized).most_common(1)[0]
            # Find an original pred with this normalized form
            for pr in preds:
                if pr and normalize(pr) == most_common:
                    if sympy_eq(pr, p["answer"]): sc_correct += 1
                    break

    result = {
        "model": args.model, "n_samples": args.n_samples,
        "greedy_first": greedy_correct,
        "oracle_pass_at_N": oracle_correct,
        "self_consistency": sc_correct,
        "n": len(math500),
        "elapsed_s": time.time() - T0,
    }
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — SELF-CONSISTENCY vs ORACLE on MATH-500 (n={args.n_samples})")
    print(f"    First sample (greedy-like): {greedy_correct}/{len(math500)} ({100*greedy_correct/len(math500):.1f}%)")
    print(f"    Self-consistency (vote):    {sc_correct}/{len(math500)} ({100*sc_correct/len(math500):.1f}%)")
    print(f"    Oracle (any-pass):          {oracle_correct}/{len(math500)} ({100*oracle_correct/len(math500):.1f}%)")
    sc_recovery = 100*(sc_correct - greedy_correct)/(oracle_correct - greedy_correct) if oracle_correct > greedy_correct else 0
    print(f"    SC recovers {sc_recovery:.0f}% of oracle-greedy gap")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
