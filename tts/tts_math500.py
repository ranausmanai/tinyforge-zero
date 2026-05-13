"""TTS on MATH-500: greedy + best-of-N pass@1.

If TTS works on math like it does on code, we should see major lift.
"""
import os, json, time, re, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset
import sympy
from sympy.parsing.latex import parse_latex

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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
        if abs(float(a) - float(b)) < 1e-6: return True
    except Exception: pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/tts_math/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
    log(f"  loaded")

    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    log(f"  MATH-500: {len(ds)} problems")

    SYS = "You are a careful math problem solver. End with \\boxed{answer}."
    USER_TEMPLATE = "Solve this competition math problem. Show your reasoning, then put the final answer in \\boxed{{...}}.\n\nProblem: {problem}\n\nSolution:"
    prompts = []
    for p in ds:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": USER_TEMPLATE.format(problem=p["problem"])}]
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(USER_TEMPLATE.format(problem=p["problem"]))

    # Greedy
    log("=== GREEDY ===")
    sp_g = SamplingParams(temperature=0, max_tokens=800)
    t0 = time.time()
    g_outs = [o.outputs[0].text for o in llm.generate(prompts, sp_g, use_tqdm=False)]
    log(f"  gen in {time.time()-t0:.1f}s")
    g_correct = sum(1 for p, raw in zip(ds, g_outs) if sympy_equal(extract_boxed(raw), p["answer"]))
    log(f"  GREEDY: {g_correct}/{len(ds)} ({100*g_correct/len(ds):.1f}%)")

    # Best-of-N (any correct)
    log(f"=== BEST-OF-{args.n_samples} (temp={args.temperature}) ===")
    sp_s = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=800, n=args.n_samples)
    t0 = time.time()
    s_outs = llm.generate(prompts, sp_s, use_tqdm=False)
    log(f"  gen in {time.time()-t0:.1f}s")
    bN_correct = 0
    for p, outset in zip(ds, s_outs):
        for o in outset.outputs:
            if sympy_equal(extract_boxed(o.text), p["answer"]):
                bN_correct += 1; break

    result = {"model": args.model, "n_samples": args.n_samples, "temperature": args.temperature,
              "greedy": g_correct, "best_of_N": bN_correct, "n": len(ds), "elapsed_s": time.time()-T0}
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — MATH-500 ({len(ds)} problems)")
    print(f"    Greedy:           {g_correct}/{len(ds)} ({100*g_correct/len(ds):.1f}%)")
    print(f"    Best-of-{args.n_samples}:        {bN_correct}/{len(ds)} ({100*bN_correct/len(ds):.1f}%)")
    print(f"    TTS Lift: +{bN_correct - g_correct} ({100*(bN_correct-g_correct)/len(ds):.1f}pp)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
