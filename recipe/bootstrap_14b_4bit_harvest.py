"""Bootstrap loop adapted for large models — uses 4-bit NF4 quantization and batch=1.
Just the harvest loop (no training during loop). Saves pairs.
"""
import os, sys, json, time, re, gc, subprocess, tempfile, argparse, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_code(text):
    if "```python" in text: text = text.split("```python", 1)[1]
    elif "```" in text: text = text.split("```", 1)[1]
    if "```" in text: text = text.split("```", 1)[0]
    return text.strip()


def run_python(code, timeout=8):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout, text=True, cwd="/tmp")
        if r.returncode == 0: return True, ""
        err = (r.stderr or r.stdout).strip().splitlines()
        return False, "\n".join(err[-3:])[:300]
    except subprocess.TimeoutExpired: return False, "timeout"
    finally:
        try: os.unlink(path)
        except: pass


def gen_one(model, tok, prompt, max_new=400, temperature=0.0):
    msgs = [{"role": "system", "content": "You are a Python coder."},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else 1.0, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


PROBLEM_GEN_PROMPT = """Generate ONE simple Python coding problem with a clear function spec and 3 test assertions.

Output format (exactly one ```python block):

```python
def {function_name}({args}):
    \"\"\"{one-line description of what the function does}\"\"\"
    {implementation}

# tests
assert {function_name}(...) == ...
assert {function_name}(...) == ...
assert {function_name}(...) == ...
```

Make the function specific and concrete. Output ONLY the code block."""


def parse_problem(raw_code):
    code = raw_code.strip()
    if "def " not in code: return None
    lines = code.split("\n")
    func_start = next((i for i, l in enumerate(lines) if l.startswith("def ")), None)
    if func_start is None: return None
    tests = []
    def_end = None
    for i in range(func_start, len(lines)):
        l = lines[i]
        if l.startswith("def ") and i > func_start: break
        if l.startswith("assert "):
            tests.append(l)
            if def_end is None: def_end = i
    if len(tests) < 2: return None
    if def_end is None: def_end = len(lines)
    full_solution = "\n".join(lines[func_start:def_end]).strip()
    if len(full_solution) < 30: return None
    m = re.match(r"def\s+(\w+)\s*\(", lines[func_start])
    if not m: return None
    sig_lines = []
    for i in range(func_start, def_end):
        sig_lines.append(lines[i])
        if i == func_start and not any('"""' in lines[j] for j in range(i, min(i+5, def_end))):
            sig_lines.append("    pass"); break
        if i > func_start and '"""' in lines[i] and ('"""' in lines[i-1] or lines[i].count('"""') >= 2):
            break
    return {"fn_name": m.group(1), "signature": "\n".join(sig_lines), "tests": tests, "canonical": full_solution}


def humaneval_full(model, tok):
    he = list(load_dataset("openai_humaneval", split="test"))
    log(f"  full HumanEval: {len(he)} problems")
    correct = 0
    for i, p in enumerate(he):
        prompt = p["prompt"] + "\n# Complete the function above."
        raw = gen_one(model, tok, prompt, max_new=400, temperature=0.0)
        code = extract_code(raw) if "```" in raw else raw
        full = p["prompt"] + "\n" + code if "def " not in code else code
        test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
        ok, _ = run_python(test_code, timeout=10)
        if ok: correct += 1
        if (i+1) % 20 == 0: log(f"    eval {i+1}/{len(he)}: {correct} correct")
    return correct, len(he)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--problems_per_iter", type=int, default=8)
    ap.add_argument("--n_attempts", type=int, default=4)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = f"/workspace/bootstrap14b/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model} in 4-bit NF4")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_cfg,
                                                  device_map="cuda:0")
    model.eval()
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    log("INITIAL eval on full HumanEval")
    base_correct, base_total = humaneval_full(model, tok)
    log(f"  base: {base_correct}/{base_total}")

    accumulated = []
    for it in range(1, args.iterations + 1):
        it_t = time.time()
        valid_problems = []
        for _ in range(args.problems_per_iter):
            raw = gen_one(model, tok, PROBLEM_GEN_PROMPT, max_new=400, temperature=0.9)
            code = extract_code(raw) if "```" in raw else raw
            parsed = parse_problem(code)
            if not parsed: continue
            full = parsed["canonical"] + "\n\n" + "\n".join(parsed["tests"])
            ok, _ = run_python(full)
            if ok: valid_problems.append(parsed)

        new_pairs = 0
        for p in valid_problems:
            attempts = []
            solve_prompt = f"Implement: {p['signature']}\n\nTests:\n{chr(10).join(p['tests'])}\n\nOutput only the function implementation in one ```python block."
            for _ in range(args.n_attempts):
                raw = gen_one(model, tok, solve_prompt, max_new=400, temperature=0.8)
                attempts.append(raw)
            broken = None; fixed = None
            for raw in attempts:
                code = extract_code(raw) if "```" in raw else raw
                full = code + "\n\n" + "\n".join(p["tests"])
                ok, err = run_python(full)
                if ok and fixed is None: fixed = code
                elif not ok and broken is None: broken = code; broken_err = err
                if broken and fixed: break
            if broken and fixed:
                accumulated.append({"signature": p["signature"], "tests": p["tests"],
                                    "broken": broken, "error": broken_err if 'broken_err' in dir() else "",
                                    "fixed": fixed})
                new_pairs += 1

        log(f"iter {it}: {len(valid_problems)} valid, {new_pairs} pairs (total: {len(accumulated)})  [{time.time()-it_t:.0f}s]")
        with open(f"{out_dir}/pairs.jsonl", "w") as fh:
            for r in accumulated: fh.write(json.dumps(r) + "\n")

    log(f"DONE — accumulated {len(accumulated)} pairs from {args.iterations} iters")
    print()
    print("=" * 70)
    print(f"  14B BASELINE: {base_correct}/{base_total} on HumanEval")
    print(f"  Accumulated pairs: {len(accumulated)}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
