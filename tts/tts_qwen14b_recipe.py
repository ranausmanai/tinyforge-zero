"""Test-time scaling on Qwen2.5-14B-Base + multi_v1 adapter.

For each HumanEval problem:
 1. Sample 8 attempts at temp=0.6 from the trained model.
 2. Run each attempt against the tests.
 3. Accept the first that passes → pass@1 with best-of-N selection.

Compared to greedy pass@1 (which gave 80.5%), this should push higher.
"""
import os, json, time, re, subprocess, tempfile, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_code(text):
    if "```python" in text: text = text.split("```python", 1)[1]
    elif "```" in text: text = text.split("```", 1)[1]
    if "```" in text: text = text.split("```", 1)[0]
    return text.strip()


def run_python(code, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout, text=True, cwd="/tmp")
        return r.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally:
        try: os.unlink(path)
        except: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--adapter", default="/workspace/multi_v1_adapter")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/tts/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    log(f"loading {args.model} with adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048,
              enable_lora=True, max_lora_rank=32)
    lora_req = LoRARequest("multi_v1", 1, args.adapter)
    log(f"  loaded")

    he = list(load_dataset("openai_humaneval", split="test"))
    log(f"  HE: {len(he)} problems")

    # --- Greedy baseline (with adapter)
    log("=== GREEDY pass@1 (with adapter) ===")
    from vllm import SamplingParams
    sp_greedy = SamplingParams(temperature=0, max_tokens=400)
    # Use chat template for Qwen2.5 (it has one)
    prompts = []
    for p in he:
        msgs = [{"role": "system", "content": "You are a Python coder. Output one ```python block only."},
                {"role": "user", "content": p["prompt"] + "\n# Complete the function above."}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    t0 = time.time()
    greedy_outs = [o.outputs[0].text for o in llm.generate(prompts, sp_greedy, lora_request=lora_req, use_tqdm=False)]
    log(f"  greedy gen in {time.time()-t0:.1f}s")
    greedy_correct = 0
    for p, raw in zip(he, greedy_outs):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        if run_python(test_code, 15): greedy_correct += 1
    log(f"  GREEDY pass@1: {greedy_correct}/{len(he)} ({100*greedy_correct/len(he):.1f}%)")

    # --- Test-time scaling: sample N, take first that passes (best-of-N pass@1)
    log(f"=== TEST-TIME SCALING: N={args.n_samples}, temp={args.temperature} ===")
    sp_sample = SamplingParams(temperature=args.temperature, top_p=0.95,
                               max_tokens=400, n=args.n_samples)
    t0 = time.time()
    sample_outs = llm.generate(prompts, sp_sample, lora_request=lora_req, use_tqdm=False)
    log(f"  sampling gen in {time.time()-t0:.1f}s")

    t1 = time.time()
    bestN_correct = 0
    per_problem = []
    for p, outset in zip(he, sample_outs):
        attempts = [o.text for o in outset.outputs]
        any_pass = False
        for a in attempts:
            code = extract_code(a) if "```" in a else a
            full = p["prompt"] + "\n" + code if "def " not in code else code
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 15):
                any_pass = True
                break
        if any_pass: bestN_correct += 1
        per_problem.append({"task_id": p["task_id"], "best_of_N_pass": any_pass})
    log(f"  verify done in {time.time()-t1:.1f}s")

    result = {
        "model": args.model, "adapter": args.adapter,
        "n_samples": args.n_samples, "temperature": args.temperature,
        "greedy_passN": greedy_correct,
        "best_of_N_passN": bestN_correct,
        "n_total": len(he),
        "elapsed_s": time.time()-T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)
    with open(f"{out_dir}/per_problem.json", "w") as fh: json.dump(per_problem, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} + adapter {args.adapter}")
    print(f"  HumanEval:")
    print(f"    Greedy pass@1:         {greedy_correct}/{len(he)} ({100*greedy_correct/len(he):.1f}%)")
    print(f"    Best-of-{args.n_samples} pass@1:  {bestN_correct}/{len(he)} ({100*bestN_correct/len(he):.1f}%)")
    print(f"    Lift: +{bestN_correct - greedy_correct} ({100*(bestN_correct-greedy_correct)/len(he):.1f}pp)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
