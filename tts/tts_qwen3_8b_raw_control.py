"""Control: Qwen3-8B-Base RAW (no recipe) + best-of-8 on HumanEval.

Tells us if the 89.6% headline on 14B+recipe is driven by recipe or by test-time scaling.
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/tts_raw/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading {args.model} (no adapter)")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
    log(f"  loaded")

    he = list(load_dataset("openai_humaneval", split="test"))
    log(f"  HE: {len(he)} problems")

    # Try chat-template style if available, else raw
    prompts = []
    for p in he:
        try:
            msgs = [{"role": "system", "content": "You are a Python coder. Output one ```python block only."},
                    {"role": "user", "content": p["prompt"] + "\n# Complete the function above."}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(p["prompt"])

    # --- Greedy
    log("=== GREEDY pass@1 ===")
    sp_g = SamplingParams(temperature=0, max_tokens=400)
    t0 = time.time()
    g_outs = [o.outputs[0].text for o in llm.generate(prompts, sp_g, use_tqdm=False)]
    log(f"  greedy gen in {time.time()-t0:.1f}s")
    g_correct = 0
    for p, raw in zip(he, g_outs):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        if run_python(test_code, 15): g_correct += 1
    log(f"  GREEDY pass@1: {g_correct}/{len(he)} ({100*g_correct/len(he):.1f}%)")

    # --- Best-of-N
    log(f"=== BEST-OF-{args.n_samples} (temp={args.temperature}) ===")
    sp_s = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=400, n=args.n_samples)
    t0 = time.time()
    s_outs = llm.generate(prompts, sp_s, use_tqdm=False)
    log(f"  sampling gen in {time.time()-t0:.1f}s")
    t1 = time.time()
    bN_correct = 0
    for p, outset in zip(he, s_outs):
        attempts = [o.text for o in outset.outputs]
        for a in attempts:
            code = extract_code(a) if "```" in a else a
            full = p["prompt"] + "\n" + code if "def " not in code else code
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 15):
                bN_correct += 1
                break
    log(f"  verify in {time.time()-t1:.1f}s")

    result = {
        "model": args.model, "n_samples": args.n_samples, "temperature": args.temperature,
        "greedy_passN": g_correct, "best_of_N_passN": bN_correct, "n_total": len(he),
        "elapsed_s": time.time()-T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} (NO ADAPTER) — HumanEval")
    print(f"    Greedy pass@1:         {g_correct}/{len(he)} ({100*g_correct/len(he):.1f}%)")
    print(f"    Best-of-{args.n_samples} pass@1:  {bN_correct}/{len(he)} ({100*bN_correct/len(he):.1f}%)")
    print(f"    Lift from TTS: +{bN_correct - g_correct} ({100*(bN_correct-g_correct)/len(he):.1f}pp)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
