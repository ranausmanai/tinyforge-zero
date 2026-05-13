"""TTS scaling sweep: pass@1 across N samples for HE + HE+ + MATH-500."""
import os, json, time, re, subprocess, tempfile, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_code(t):
    if "```python" in t: t = t.split("```python", 1)[1]
    elif "```" in t: t = t.split("```", 1)[1]
    if "```" in t: t = t.split("```", 1)[0]
    return t.strip()


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

    he = list(load_dataset("openai_humaneval", split="test"))
    math500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[:200]

    # Build prompts
    he_prompts = []
    for p in he:
        try:
            msgs = [{"role": "system", "content": "You are a Python coder. Output one ```python block only."},
                    {"role": "user", "content": p["prompt"] + "\n# Complete the function above."}]
            he_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            he_prompts.append(p["prompt"])

    math_prompts = []
    UTMPL = "Solve this competition math problem. End with \\boxed{{...}}.\n\nProblem: {p}\n\nSolution:"
    for p in math500:
        try:
            msgs = [{"role": "system", "content": "Math solver. End with \\boxed{answer}."},
                    {"role": "user", "content": UTMPL.format(p=p["problem"])}]
            math_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            math_prompts.append(UTMPL.format(p=p["problem"]))

    # Generate max-N samples ONCE per task (N=32), then compute pass@k for k ∈ {1, 2, 4, 8, 16, 32}
    MAX_N = 32
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=600, n=MAX_N)
    log(f"generating MAX_N={MAX_N} samples per task")
    t0 = time.time()
    he_outs = llm.generate(he_prompts, sp, use_tqdm=False)
    log(f"  HE gen in {time.time()-t0:.1f}s")
    t0 = time.time()
    math_outs = llm.generate(math_prompts, sp, use_tqdm=False)
    log(f"  MATH gen in {time.time()-t0:.1f}s")

    # Compute correctness for each sample
    def he_correct(p, raw):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        return run_python(test_code, 10)

    log("verifying HE samples...")
    he_results = []  # per task: list of bool
    for p, outset in zip(he, he_outs):
        per_task = []
        for o in outset.outputs:
            per_task.append(he_correct(p, o.text))
        he_results.append(per_task)
    log(f"  HE verify done")

    import sympy
    from sympy.parsing.latex import parse_latex
    def sympy_eq(a, b):
        if a is None or b is None: return False
        a, b = a.strip(), b.strip()
        if a == b: return True
        try:
            if sympy.simplify(parse_latex(a) - parse_latex(b)) == 0: return True
        except Exception: pass
        try:
            if abs(float(a) - float(b)) < 1e-6: return True
        except Exception: pass
        return False

    log("verifying MATH samples...")
    math_results = []
    for p, outset in zip(math500, math_outs):
        per_task = []
        for o in outset.outputs:
            pred = extract_boxed(o.text)
            per_task.append(sympy_eq(pred, p["answer"]))
        math_results.append(per_task)
    log(f"  MATH verify done")

    # Compute pass@k for each k
    NS = [1, 2, 4, 8, 16, 32]
    def best_of_k(results, k):
        return sum(1 for r in results if any(r[:k]))

    he_scaling = {k: best_of_k(he_results, k) for k in NS}
    math_scaling = {k: best_of_k(math_results, k) for k in NS}

    result = {
        "model": args.model, "tag": args.tag, "MAX_N": MAX_N,
        "humaneval_total": len(he),
        "math500_total": len(math500),
        "he_pass_at_k": he_scaling,
        "math500_pass_at_k": math_scaling,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — TTS SCALING SWEEP")
    print(f"  N    HE       MATH-500")
    for k in NS:
        print(f"  {k:>3}  {he_scaling[k]:>3}/{len(he)} ({100*he_scaling[k]/len(he):.1f}%)   "
              f"{math_scaling[k]:>3}/{len(math500)} ({100*math_scaling[k]/len(math500):.1f}%)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
