"""Recursive self-bootstrap: iter1->iter2->iter3.

Iter k:
 - Use model from previous iter (or base for iter 1)
 - Mine pairs on MBPP-train
 - Train fresh LoRA from BASE on accumulated pairs
 - Eval on HE
"""
import os, json, time, re, subprocess, tempfile, argparse, gc, random
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from datasets import load_dataset

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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


def mbpp_prompt(p):
    return f"# Task: {p['prompt']}\n# Tests:\n# " + "\n# ".join(p["test_list"]) + "\n\n"


def he_prompt(p): return p["prompt"]


def vllm_gen(llm, prompts, max_new=400, temperature=0.0, n=1, lora_req=None, stops=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                       max_tokens=max_new, n=n,
                       stop=stops or ["\nclass Test", "\nif __name__", "\n\nprint", "\nassert "])
    if lora_req:
        out = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
    else:
        out = llm.generate(prompts, sp, use_tqdm=False)
    if n == 1: return [o.outputs[0].text for o in out]
    return [[c.text for c in o.outputs] for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_iters", type=int, default=3)
    ap.add_argument("--n_mining", type=int, default=200)
    ap.add_argument("--attempts_per", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    log(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    he = list(load_dataset("openai_humaneval", split="test"))
    mbpp_full = list(load_dataset("mbpp", split="train"))
    random.seed(42); random.shuffle(mbpp_full)
    seeds_pool = []
    for p in mbpp_full[:args.n_mining * args.n_iters]:
        prompt_text = p.get("prompt") or p.get("text", "")
        if prompt_text and p.get("test_list"):
            seeds_pool.append({"prompt": prompt_text, "test_list": p["test_list"]})
    log(f"seeds pool: {len(seeds_pool)}")

    iter_results = []
    accumulated_pairs = []
    current_adapter = None  # path

    for it in range(1, args.n_iters + 1):
        log(f"\n========== ITER {it} ==========")
        # Load model (with current adapter if exists)
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=2048,
                  enable_lora=(current_adapter is not None), max_lora_rank=16)
        lora_req = LoRARequest("cur", 1, current_adapter) if current_adapter else None
        log(f"  loaded {'(with adapter)' if current_adapter else '(base)'}")

        # Mine pairs using current model
        seeds = seeds_pool[(it-1)*args.n_mining:it*args.n_mining]
        log(f"  mining from {len(seeds)} new seeds")
        prompts = [mbpp_prompt(p) for p in seeds]
        greedy_outs = vllm_gen(llm, prompts, max_new=400, lora_req=lora_req)
        hard_idx = []
        for i, (p, raw) in enumerate(zip(seeds, greedy_outs)):
            test_code = raw + "\n\n" + "\n".join(p["test_list"])
            if not run_python(test_code, 8):
                hard_idx.append(i)
        log(f"  greedy: {len(seeds)-len(hard_idx)} pass, {len(hard_idx)} hard")

        if hard_idx:
            hard_prompts = [mbpp_prompt(seeds[i]) for i in hard_idx]
            sample_outs = vllm_gen(llm, hard_prompts, max_new=400, temperature=0.8,
                                    n=args.attempts_per, lora_req=lora_req)
            new_pairs = []
            for j, i in enumerate(hard_idx):
                attempts = sample_outs[j]
                passes = []
                for a in attempts:
                    if run_python(a + "\n\n" + "\n".join(seeds[i]["test_list"]), 8):
                        passes.append(a); break
                if passes:
                    new_pairs.append({"problem": seeds[i]["prompt"], "tests": seeds[i]["test_list"],
                                       "broken": greedy_outs[i].strip(), "fixed": passes[0].strip(),
                                       "iter": it})
            accumulated_pairs.extend(new_pairs)
            log(f"  mined {len(new_pairs)} new pairs (cumulative: {len(accumulated_pairs)})")

        # Eval current model on HE
        log(f"  eval HE...")
        he_outs = vllm_gen(llm, [he_prompt(p) for p in he], max_new=400, lora_req=lora_req,
                          stops=["\nclass ", "\nif __name__", "\n\nprint"])
        he_correct = 0
        for p, raw in zip(he, he_outs):
            full = p["prompt"] + "\n" + raw
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 10): he_correct += 1
        log(f"  HE iter{it} (pre-train): {he_correct}/{len(he)}")
        iter_results.append({"iter": it, "he_pretrain": he_correct, "cumulative_pairs": len(accumulated_pairs)})

        # Tear down vLLM, train new adapter on accumulated pairs
        del llm; gc.collect(); torch.cuda.empty_cache()

        if len(accumulated_pairs) < 5:
            log(f"  too few pairs to train, skipping iter {it} training")
            continue

        from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
        from datasets import Dataset as HFDataset
        from peft import LoraConfig, get_peft_model

        def mk_ex(r):
            user = (f"# Task: {r['problem']}\n# Tests:\n# " + "\n# ".join(r['tests']) + "\n"
                    f"# My broken attempt:\n{r['broken']}\n# Corrected:\n")
            target = r["fixed"]
            full = user + target
            full_ids = tok(full, add_special_tokens=False)["input_ids"]
            user_ids = tok(user, add_special_tokens=False)["input_ids"]
            MAX = 1024
            full_ids = full_ids[:MAX]
            labels = list(full_ids)
            n_user = min(len(user_ids), len(labels))
            for i in range(n_user): labels[i] = -100
            pad = MAX - len(full_ids)
            return {"input_ids": full_ids + [tok.pad_token_id]*pad,
                    "attention_mask": [1]*len(full_ids) + [0]*pad,
                    "labels": labels + [-100]*pad}

        log(f"  training fresh adapter on {len(accumulated_pairs)} pairs...")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        ds_train = HFDataset.from_list([mk_ex(r) for r in accumulated_pairs])
        targs = TrainingArguments(
            output_dir=f"{args.out_dir}/iter{it}_ckpt", num_train_epochs=2,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=1e-4, bf16=True, logging_steps=20,
            save_strategy="no", report_to="none", remove_unused_columns=False, warmup_ratio=0.05,
        )
        Trainer(model=model, args=targs, train_dataset=ds_train, tokenizer=tok).train()
        adapter_dir = f"{args.out_dir}/iter{it}_adapter"
        model.save_pretrained(adapter_dir)
        del model; gc.collect(); torch.cuda.empty_cache()
        current_adapter = adapter_dir

        # Re-eval with new adapter to get post-train HE
        log(f"  eval post-train HE...")
        llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
                  enable_lora=True, max_lora_rank=16)
        lora_req = LoRARequest(f"iter{it}", it, current_adapter)
        he_outs = vllm_gen(llm, [he_prompt(p) for p in he], max_new=400, lora_req=lora_req,
                          stops=["\nclass ", "\nif __name__", "\n\nprint"])
        he_correct = 0
        for p, raw in zip(he, he_outs):
            full = p["prompt"] + "\n" + raw
            test_code = full + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})"
            if run_python(test_code, 10): he_correct += 1
        log(f"  HE iter{it} (post-train): {he_correct}/{len(he)}")
        iter_results[-1]["he_posttrain"] = he_correct

        del llm; gc.collect(); torch.cuda.empty_cache()

    # Save pairs and results
    with open(f"{args.out_dir}/pairs.jsonl", "w") as fh:
        for r in accumulated_pairs: fh.write(json.dumps(r) + "\n")
    result = {"model": args.model, "tag": args.tag, "n_iters": args.n_iters,
              "iter_results": iter_results, "total_pairs": len(accumulated_pairs),
              "elapsed_s": time.time() - T0}
    with open(f"{args.out_dir}/result.json", "w") as fh: json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  {args.model} — RECURSIVE BOOTSTRAP")
    for r in iter_results:
        pre = r.get("he_pretrain", "-")
        post = r.get("he_posttrain", "-")
        print(f"  iter {r['iter']}: cum_pairs={r['cumulative_pairs']} HE_pre={pre} HE_post={post}")
    print(f"  Time: {time.time()-T0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
