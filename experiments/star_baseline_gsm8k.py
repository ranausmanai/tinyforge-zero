"""STaR / Rejection Sampling Fine-Tuning on GSM8K.

For each GSM8K-train problem:
  - sample N reasoning chains at temp=0.8
  - keep chains that produce correct final answer
  - train on (problem, correct chain) pairs
Then eval on GSM8K-test.
"""
import os, sys, json, time, re, gc, argparse, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def extract_answer(text: str):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m: return float(m.group(1))
    m = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", text)
    if m: return float(m.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        try: return float(matches[-1])
        except: return None
    return None


def gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=8):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i+batch]
        texts = []
        for p in chunk:
            msgs = [{"role": "system", "content": "You are a careful math tutor."},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else 1.0, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        for j in range(out.size(0)):
            outs.append(tok.decode(out[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


SOLVE_PROMPT = "Solve this math problem step by step. End with the answer on a new line as: #### <number>\n\nProblem: {problem}"


def parse_gold(answer_field: str):
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_field)
    return float(m.group(1).replace(",", "")) if m else None


def gsm8k_eval(model, tok, n=200):
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))[:n]
    log(f"  eval on GSM8K-test ({len(ds)} problems)")
    prompts = [SOLVE_PROMPT.format(problem=p["question"]) for p in ds]
    outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.0, batch=8)
    correct = 0
    for p, raw in zip(ds, outs):
        gold = parse_gold(p["answer"])
        if gold is None: continue
        pred = extract_answer(raw)
        if pred is not None and abs(pred - gold) < 0.01: correct += 1
    return correct, len(ds)


def make_train_example(problem: str, solution: str, tok):
    user = SOLVE_PROMPT.format(problem=problem)
    msgs_pre = [{"role": "system", "content": "You are a careful math tutor."},
                {"role": "user", "content": user}]
    msgs_full = msgs_pre + [{"role": "assistant", "content": solution}]
    pre = tok.apply_chat_template(msgs_pre, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs_full, tokenize=False)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    MAX = 1024
    full_ids = full_ids[:MAX]
    labels = list(full_ids)
    n_pre = min(len(pre_ids), len(labels))
    for i in range(n_pre): labels[i] = -100
    pad = MAX - len(full_ids)
    return {"input_ids": full_ids + [tok.pad_token_id]*pad,
            "attention_mask": [1]*len(full_ids) + [0]*pad,
            "labels": labels + [-100]*pad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--n_train_problems", type=int, default=300)
    ap.add_argument("--n_chains", type=int, default=8)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = f"/workspace/star/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    log(f"  loaded mem={torch.cuda.memory_allocated('cuda:0')/1e9:.1f}GB")

    # Initial eval on GSM8K-test
    model.eval()
    log("INITIAL eval on GSM8K-test")
    base_correct, base_total = gsm8k_eval(model, tok, n=args.n_eval)
    log(f"  GSM8K-test base: {base_correct}/{base_total}")

    # Mine reasoning chains from GSM8K-train
    log(f"mining reasoning chains from GSM8K-train ({args.n_train_problems} problems × {args.n_chains} chains)")
    train_set = list(load_dataset("openai/gsm8k", "main", split="train"))[:args.n_train_problems]
    pairs = []
    BATCH_PROBLEMS = 8  # batch problems together
    for batch_start in range(0, len(train_set), BATCH_PROBLEMS):
        batch_end = min(batch_start + BATCH_PROBLEMS, len(train_set))
        batch_problems = train_set[batch_start:batch_end]
        # For each problem, generate N chains. So total = batch_size * N
        prompts = []
        for p in batch_problems:
            for _ in range(args.n_chains):
                prompts.append(SOLVE_PROMPT.format(problem=p["question"]))
        outs = gen_batch(model, tok, prompts, max_new=400, temperature=0.8, batch=8)
        # Outs are in problem-major × chain-major order
        for i, p in enumerate(batch_problems):
            gold = parse_gold(p["answer"])
            if gold is None: continue
            chain_outs = outs[i*args.n_chains : (i+1)*args.n_chains]
            for raw in chain_outs:
                pred = extract_answer(raw)
                if pred is not None and abs(pred - gold) < 0.01:
                    pairs.append({"problem": p["question"], "solution": raw.strip()})
                    break  # take first correct chain per problem
        log(f"  mined {len(pairs)} pairs from {batch_end} problems")

    if not pairs:
        log("FATAL: no pairs mined")
        return
    with open(f"{out_dir}/pairs.jsonl", "w") as fh:
        for p in pairs: fh.write(json.dumps(p) + "\n")
    log(f"total pairs mined: {len(pairs)} from {len(train_set)} problems "
        f"(coverage: {len(pairs)/len(train_set)*100:.1f}%)")

    # Train
    log(f"TRAINING on {len(pairs)} pairs, {args.epochs} epochs")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    tok.padding_side = "right"
    ds = HFDataset.from_list([make_train_example(p["problem"], p["solution"], tok) for p in pairs])
    targs = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, bf16=True, logging_steps=20,
        save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
    )
    Trainer(model=model, args=targs, train_dataset=ds, processing_class=tok).train()
    log("training done")
    tok.padding_side = "left"

    # Final eval
    model.eval()
    log("FINAL eval on GSM8K-test")
    trained_correct, trained_total = gsm8k_eval(model, tok, n=args.n_eval)
    log(f"  GSM8K-test trained: {trained_correct}/{trained_total}")

    result = {
        "model": args.model, "n_train_problems": args.n_train_problems,
        "n_chains": args.n_chains, "n_pairs_mined": len(pairs),
        "epochs": args.epochs, "seed": args.seed,
        "base": [base_correct, base_total],
        "trained": [trained_correct, trained_total],
        "delta": trained_correct - base_correct,
        "elapsed_s": time.time() - T0,
    }
    with open(f"{out_dir}/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  STaR / RFT on GSM8K — {args.model}")
    print(f"  Mined {len(pairs)} pairs from {len(train_set)} GSM8K-train problems ({len(pairs)/len(train_set)*100:.1f}% coverage)")
    print(f"  GSM8K-test:  base={base_correct}/{base_total}  trained={trained_correct}/{trained_total}  Δ={trained_correct-base_correct:+d}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
