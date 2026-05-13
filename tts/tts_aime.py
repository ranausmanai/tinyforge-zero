"""TTS on AIME (Olympiad math). 90 problems, integer answers 0-999.
If 8B+best-of-N hits 30%+, that's matching frontier reasoning models."""
import os, json, time, re, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_int(text):
    """AIME answers are integers 0-999. Try \boxed first, fall back to last integer."""
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        try: return int(m.group(1))
        except: return None
    # Last integer in last few lines
    lines = text.strip().split("\n")
    for line in reversed(lines[-5:]):
        nums = re.findall(r"\b(\d+)\b", line)
        if nums:
            try: return int(nums[-1])
            except: pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/tts_aime/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=3072)
    log(f"  loaded")

    ds = list(load_dataset("AI-MO/aimo-validation-aime", split="train"))
    log(f"  AIME: {len(ds)} problems")

    SYS = "You are a careful math problem solver. AIME answers are integers between 0 and 999. End with \\boxed{integer}."
    UTMPL = "Solve this AIME problem. Show your reasoning, then put the final integer answer in \\boxed{{...}}.\n\nProblem: {problem}\n\nSolution:"
    prompts = []
    for p in ds:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": UTMPL.format(problem=p["problem"])}]
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(UTMPL.format(problem=p["problem"]))

    log("=== GREEDY ===")
    sp_g = SamplingParams(temperature=0, max_tokens=2000)
    t0 = time.time()
    g_outs = [o.outputs[0].text for o in llm.generate(prompts, sp_g, use_tqdm=False)]
    log(f"  gen in {time.time()-t0:.1f}s")
    g_correct = 0
    for p, raw in zip(ds, g_outs):
        pred = extract_int(raw)
        gold = int(p["answer"])
        if pred == gold: g_correct += 1
    log(f"  GREEDY: {g_correct}/{len(ds)} ({100*g_correct/len(ds):.1f}%)")

    log(f"=== BEST-OF-{args.n_samples} (temp={args.temperature}) ===")
    sp_s = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=2000, n=args.n_samples)
    t0 = time.time()
    s_outs = llm.generate(prompts, sp_s, use_tqdm=False)
    log(f"  gen in {time.time()-t0:.1f}s")
    bN_correct = 0
    for p, outset in zip(ds, s_outs):
        gold = int(p["answer"])
        for o in outset.outputs:
            pred = extract_int(o.text)
            if pred == gold:
                bN_correct += 1; break

    result = {"model": args.model, "n_samples": args.n_samples, "temperature": args.temperature,
              "greedy": g_correct, "best_of_N": bN_correct, "n": len(ds), "elapsed_s": time.time()-T0}
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — AIME ({len(ds)} problems)")
    print(f"    Greedy:        {g_correct}/{len(ds)} ({100*g_correct/len(ds):.1f}%)")
    print(f"    Best-of-{args.n_samples}:     {bN_correct}/{len(ds)} ({100*bN_correct/len(ds):.1f}%)")
    print(f"    TTS Lift: +{bN_correct - g_correct} ({100*(bN_correct-g_correct)/len(ds):.1f}pp)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
