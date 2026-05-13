"""Eval our best 14B adapter on HumanEval+ (contamination-resistant hidden tests)."""
import os, json, time, re, subprocess, tempfile, argparse
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

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


def gen_batch(model, tok, prompts, max_new=400, batch=4):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "system", "content": "You are a Python coder. Output one ```python block only."},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for j in range(out.size(0)):
            outs.append(tok.decode(out[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--adapter", default="/workspace/multi_pair/multi_v1/adapter")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/eval_plus/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    if args.adapter and os.path.exists(args.adapter):
        log(f"  loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        log("  no adapter — base only")
    model.eval()

    # Load HumanEval+ via evalplus dataset
    log("loading HumanEvalPlus dataset")
    ds = list(load_dataset("evalplus/humanevalplus", split="test"))
    log(f"  {len(ds)} problems")

    # Eval
    log("eval...")
    prompts = [p["prompt"] + "\n# Complete the function above." for p in ds]
    outs = gen_batch(model, tok, prompts, max_new=400, batch=4)

    base_pass, plus_pass = 0, 0
    for i, (p, raw) in enumerate(zip(ds, outs)):
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        # Public tests
        base_test = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        b = run_python(base_test, timeout=15)
        # Plus tests (hidden harder)
        plus_check = p.get("plus_input", None)
        if plus_check is not None and "plus_test" in p:
            plus_test = full + "\n\n" + p["plus_test"] + f"\n\ncheck({p['entry_point']})"
            pp = run_python(plus_test, timeout=15)
        else:
            pp = b  # fallback
        if b: base_pass += 1
        if pp: plus_pass += 1
        if (i+1) % 20 == 0:
            log(f"  {i+1}/{len(ds)}: base={base_pass}, plus={plus_pass}")

    result = {"model": args.model, "adapter": args.adapter,
              "base_pass": base_pass, "plus_pass": plus_pass, "n": len(ds),
              "elapsed_s": time.time() - T0}
    with open(f"{out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  HumanEval+   public: {base_pass}/{len(ds)}   plus(hidden): {plus_pass}/{len(ds)}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
