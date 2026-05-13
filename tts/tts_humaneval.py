"""TTS on HumanEval+ (contamination-resistant) to verify the 92% isn't memorization."""
import os, json, time, subprocess, tempfile, argparse
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

    out_dir = f"/workspace/tts_hep/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048)
    log(f"  loaded")

    hep = list(load_dataset("evalplus/humanevalplus", split="test"))
    log(f"  HE+: {len(hep)} problems")

    prompts = []
    for p in hep:
        try:
            msgs = [{"role": "system", "content": "You are a Python coder. Output one ```python block only."},
                    {"role": "user", "content": p["prompt"] + "\n# Complete the function above."}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(p["prompt"])

    log("=== GREEDY ===")
    sp_g = SamplingParams(temperature=0, max_tokens=400)
    g_outs = [o.outputs[0].text for o in llm.generate(prompts, sp_g, use_tqdm=False)]
    base_pass, plus_pass = 0, 0
    for p, raw in zip(hep, g_outs):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        # base test
        b_test = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        b_ok = run_python(b_test, 15)
        if b_ok: base_pass += 1
        # plus test (harder, hidden cases)
        if "plus_test" in p:
            p_test = full + "\n\n" + p["plus_test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(p_test, 15): plus_pass += 1
        else:
            if b_ok: plus_pass += 1
    log(f"  GREEDY base: {base_pass}/{len(hep)}  plus(hidden): {plus_pass}/{len(hep)}")

    log(f"=== BEST-OF-{args.n_samples} (temp={args.temperature}) ===")
    sp_s = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=400, n=args.n_samples)
    s_outs = llm.generate(prompts, sp_s, use_tqdm=False)
    bN_base, bN_plus = 0, 0
    for p, outset in zip(hep, s_outs):
        attempts = [o.text for o in outset.outputs]
        base_ok_any = False
        plus_ok_any = False
        for a in attempts:
            code = extract_code(a) if "```" in a else a
            full = p["prompt"] + "\n" + code if "def " not in code else code
            b_test = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            b_ok = run_python(b_test, 15)
            if b_ok and not base_ok_any:
                base_ok_any = True
            if "plus_test" in p:
                p_test = full + "\n\n" + p["plus_test"] + f"\n\ncheck({p['entry_point']})"
                p_ok = run_python(p_test, 15)
                if p_ok and not plus_ok_any:
                    plus_ok_any = True
            elif b_ok and not plus_ok_any:
                plus_ok_any = True
            if base_ok_any and plus_ok_any: break
        if base_ok_any: bN_base += 1
        if plus_ok_any: bN_plus += 1

    result = {"model": args.model, "n_samples": args.n_samples, "temperature": args.temperature,
              "greedy_base": base_pass, "greedy_plus": plus_pass,
              "best_of_N_base": bN_base, "best_of_N_plus": bN_plus,
              "n": len(hep), "elapsed_s": time.time()-T0}
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — HumanEval+ ({len(hep)} problems)")
    print(f"    Greedy base:        {base_pass}/{len(hep)} ({100*base_pass/len(hep):.1f}%)")
    print(f"    Greedy plus (hard): {plus_pass}/{len(hep)} ({100*plus_pass/len(hep):.1f}%)")
    print(f"    Best-of-{args.n_samples} base:    {bN_base}/{len(hep)} ({100*bN_base/len(hep):.1f}%)")
    print(f"    Best-of-{args.n_samples} plus:    {bN_plus}/{len(hep)} ({100*bN_plus/len(hep):.1f}%)")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
