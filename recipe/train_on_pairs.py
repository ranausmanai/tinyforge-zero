"""Train a LoRA adapter on a released `pairs.jsonl` file and evaluate.

This is the clean replication entrypoint: skip the mining stage entirely
and just consume the (broken, fixed) pairs we already mined and released
in data/. Use this to reproduce the headline numbers without burning
GPU on the search step.

Schema of pairs.jsonl (one JSON object per line):
    {
      "signature": "def foo(x): ...",      # function header + docstring
      "tests":     ["assert foo(1) == 2", ...],
      "broken":    "def foo(x): ... # buggy",
      "error":     "AssertionError ...",
      "fixed":     "def foo(x): ... # correct"
    }

Example:
    python recipe/train_on_pairs.py \\
        --model Qwen/Qwen2.5-7B \\
        --pairs data/pairs_7b_40.jsonl \\
        --out  adapter_7b_seed13 \\
        --seed 13

Then evaluate the resulting adapter with:
    python recipe/eval_raw.py --model Qwen/Qwen2.5-7B \\
        --adapter adapter_7b_seed13 --bench humaneval
"""
import argparse, json, os, random, time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPAIR_PROMPT = """### Task
Fix the bug in the Python function so it passes all the provided tests.

### Tests
{tests}

### Buggy code
```python
{broken}
```

### Error
{error}

### Fixed code
```python
{fixed}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. Qwen/Qwen2.5-7B")
    ap.add_argument("--pairs", required=True,
                    help="Path to a pairs.jsonl file (one JSON object per line)")
    ap.add_argument("--out", required=True,
                    help="Output directory for the trained LoRA adapter")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log(f"Loading pairs from {args.pairs}")
    pairs = [json.loads(l) for l in open(args.pairs)]
    log(f"  {len(pairs)} pairs")

    log(f"Loading tokenizer + base model {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    log(f"Attaching LoRA (rank {args.lora_rank}, q/k/v/o projections)")
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def format_pair(p):
        tests = "\n".join(p["tests"])
        text = REPAIR_PROMPT.format(
            tests=tests, broken=p["broken"],
            error=p.get("error", ""), fixed=p["fixed"],
        )
        ids = tok(text, truncation=True, max_length=args.max_length,
                  padding="max_length", return_tensors="pt")
        return {
            "input_ids": ids.input_ids[0],
            "attention_mask": ids.attention_mask[0],
            "labels": ids.input_ids[0].clone(),
        }

    ds = Dataset.from_list([format_pair(p) for p in pairs])

    log("Training")
    targs = TrainingArguments(
        output_dir=args.out + "_ckpt",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        seed=args.seed,
    )
    Trainer(model=model, args=targs, train_dataset=ds).train()

    log(f"Saving adapter to {args.out}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    log("Done. Evaluate with: python recipe/eval_raw.py --model "
        f"{args.model} --adapter {args.out} --bench humaneval")


if __name__ == "__main__":
    main()
