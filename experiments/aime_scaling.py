"""TTS scaling on AIME — pass@k curve from k=1 to k=64."""
import os, json, time, re, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_int(text):
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        try: return int(m.group(1))
        except: return None
    nums = re.findall(r"\b(\d+)\b", text.strip().split("\n")[-3:][-1] if text.strip().split("\n") else "")
    if nums:
        try: return int(nums[-1])
        except: pass
    return None


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
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=3072)
    log("loaded")

    ds = list(load_dataset("AI-MO/aimo-validation-aime", split="train"))
    log(f"  AIME: {len(ds)} problems")

    UTMPL = "Solve this AIME problem. Answer is integer 0-999. End with \\boxed{{N}}.\n\nProblem: {p}\n\nSolution:"
    prompts = []
    for p in ds:
        try:
            msgs = [{"role": "system", "content": "AIME solver. End with \\boxed{integer}."},
                    {"role": "user", "content": UTMPL.format(p=p["problem"])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(UTMPL.format(p=p["problem"]))

    MAX_N = 64
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1500, n=MAX_N)
    log(f"generating {MAX_N} samples per problem...")
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    log(f"  gen in {time.time()-t0:.1f}s")

    # Per-task per-sample correctness
    per_task_results = []
    for p, outset in zip(ds, outs):
        gold = int(p["answer"])
        per_sample = []
        for o in outset.outputs:
            pred = extract_int(o.text)
            per_sample.append(pred == gold)
        per_task_results.append(per_sample)

    NS = [1, 2, 4, 8, 16, 32, 64]
    scaling = {}
    for k in NS:
        scaling[k] = sum(1 for r in per_task_results if any(r[:k]))

    result = {"model": args.model, "tag": args.tag, "MAX_N": MAX_N,
              "n_total": len(ds), "pass_at_k": scaling, "elapsed_s": time.time() - T0}
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — AIME TTS SCALING")
    for k in NS:
        print(f"    pass@{k:<3}: {scaling[k]:>3}/{len(ds)} ({100*scaling[k]/len(ds):.1f}%)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
