# Reproduction Guide

Maps every paper claim → exact command. There are **two replication paths**:

- **Fast path** — use `recipe/train_on_pairs.py` with the released `data/*.jsonl`. Skips the mining stage. Gets you the trained adapter and the headline number in ~30 min on an H100.
- **Full path** — re-run the original research scripts (`bootstrap.py`, `multi_pair_14b.py`, `curriculum_math.py`) end-to-end including the self-mining step. This reproduces the recipe from scratch and verifies the mining is deterministic-ish (modulo sampling).

The fast path is what you want for paper verification. The full path is what you want if you're trying the recipe on a *new* base model.

---

## Environment

Tested on:
- **H100 80GB** (recommended for 14B runs) — Debian 12, CUDA 12.4, driver 570+
- **RTX 6000 Ada 48GB** — sufficient for 7B and 3B runs

```bash
pip install -r requirements.txt
```

Exact stack used in the paper: `torch==2.6.0`, `transformers==4.51.3`, `vllm==0.8.5`, `peft==0.13.0`.

---

## FAST PATH — reproduce headline numbers from released pairs

### Qwen2.5-7B-Base → 25 → 95–112/164 (3-seed range)

```bash
# 1. Baseline (raw-completion eval)
python recipe/eval_raw.py --model Qwen/Qwen2.5-7B --bench humaneval
# Expected: 25/164

# 2. Train on the released 40 pairs (try multiple seeds — small-data variance)
for SEED in 7 13 42; do
    python recipe/train_on_pairs.py \
        --model Qwen/Qwen2.5-7B \
        --pairs data/pairs_7b_40.jsonl \
        --out adapter_7b_seed${SEED} \
        --seed ${SEED} --lora-rank 16 --epochs 2 --lr 1e-4
    python recipe/eval_raw.py \
        --model Qwen/Qwen2.5-7B \
        --adapter adapter_7b_seed${SEED} \
        --bench humaneval
done
# Expected: seed 7 → 104/164, seed 13 → 112/164, seed 42 → 95/164
```

### Qwen2.5-14B-Base → 132/164 (80.5%) and HumanEval+ 122/164 (74.4%)

The 14B run uses 100 pairs total: the 40 warmup pairs + 60 new aggressive-mined pairs. Concatenate first, then train.

```bash
cat data/pairs_7b_40.jsonl data/pairs_14b_multi_new60.jsonl > /tmp/pairs_14b_100.jsonl

python recipe/train_on_pairs.py \
    --model Qwen/Qwen2.5-14B \
    --pairs /tmp/pairs_14b_100.jsonl \
    --out adapter_14b_multi \
    --lora-rank 32 --epochs 2 --lr 1e-4

python recipe/eval_raw.py \
    --model Qwen/Qwen2.5-14B \
    --adapter adapter_14b_multi \
    --bench humaneval
# Expected: 132/164 (80.5%) in the multi-pair eval format

python recipe/eval_plus.py \
    --model Qwen/Qwen2.5-14B \
    --adapter adapter_14b_multi
# Expected: HumanEval+ 122/164 (74.4%)
```

### Qwen2.5-3B-Base → GSM8K 32 → 66

```bash
python recipe/train_on_pairs.py \
    --model Qwen/Qwen2.5-3B \
    --pairs data/pairs_math_13.jsonl \
    --out adapter_3b_math \
    --lora-rank 16 --epochs 2 --lr 1e-4

# GSM8K eval — uses sympy as the verifier (no oracle math model needed).
# eval_raw.py auto-detects GSM8K format and runs the right verifier.
python recipe/eval_raw.py \
    --model Qwen/Qwen2.5-3B \
    --adapter adapter_3b_math \
    --bench gsm8k
# Expected: 66/100
```

---

## FULL PATH — re-mine from scratch

These reproduce the *mining* step too. Each script does generation → solving → mining → training → eval as one pipeline. They write a `pairs.jsonl` and a `result.json` under `--tag`.

### Self-bootstrap from scratch on Qwen2.5-7B

```bash
python recipe/bootstrap.py \
    --model Qwen/Qwen2.5-7B \
    --iterations 20 \
    --problems_per_iter 16 \
    --train_every 10 \
    --eval_every 10 \
    --tag bs_7b_rerun
# Writes: results/bs_7b_rerun/{pairs.jsonl,ckpt_iter*,eval_log.json,result.json}
# Expected final eval: 25 → 95–112 (seed-dependent)
```

### Aggressive multi-pair mining on Qwen2.5-14B (the 80.5% headline)

```bash
python recipe/multi_pair_14b.py \
    --model Qwen/Qwen2.5-14B \
    --warmup_pairs_path data/pairs_7b_40.jsonl \
    --n_warmup_pairs 40 \
    --n_problems 200 \
    --n_attempts 8 \
    --max_pairs_per_problem 4 \
    --lora_rank 32 --epochs 2 --lr 1e-4 \
    --tag multi_rerun
# Writes: results/multi_pair/multi_rerun/{pairs_new.jsonl,adapter/,result.json}
# Expected: trained 130–134/164 (~80%)
```

### GSM8K auto-difficulty curriculum on Qwen2.5-3B

```bash
python recipe/curriculum_math.py \
    --model Qwen/Qwen2.5-3B \
    --iterations 16 \
    --tag curr_3b_rerun
# Mines 10–15 curriculum-difficulty pairs, trains, evals.
# Expected: GSM8K 32 → 60–70 (some seed variance)
```

---

## Control experiment (Figure 2)

Verifies the signal is in the *content* of self-mined pairs, not the format. Replaces the mined pairs with mechanically-corrupted external pairs (MBPP-style) that look identical structurally.

```bash
python controls/mbpp_corrupt_control.py \
    --model Qwen/Qwen2.5-7B \
    --tag mbpp_corrupt_control
# Expected: HumanEval stays at 25/164 (Δ ≈ 0, ± seed noise)
```

---

## Pair-count sweep (Figure 3)

```bash
for N in 10 21 40; do
    head -n $N data/pairs_7b_40.jsonl > /tmp/pairs_$N.jsonl
    python recipe/train_on_pairs.py \
        --model Qwen/Qwen2.5-7B \
        --pairs /tmp/pairs_$N.jsonl \
        --out adapter_n$N --epochs 2
    python recipe/eval_raw.py \
        --model Qwen/Qwen2.5-7B --adapter adapter_n$N --bench humaneval
done
# Expected: n=10 → ~51, n=21 → 86–95, n=40 → 95–112 (seed-dependent for small N)
```

---

## Boundary conditions to verify (paper §3)

| Claim | Hint | Expected |
|-------|------|----------|
| Qwen3-8B saturated on HE | Run multi_pair_14b.py with `--model Qwen/Qwen3-8B-Base` | Base 132, adapter ≈ 118–133 — no clean lift |
| Qwen2.5-72B saturated | Same on 72B with 10 pairs | Base 83 → trained 73 (−10) |
| MATH-500 distribution mismatch | Mining on simple problems + MATH-500 eval | Base 279/500 → trained 239/500 (−40) |
| Self-correction over-correction | Train on wrong→fix triples only, no right→stays-right | Base 299/500 → trained 69/500 (−230) |
| BCB-Hard distribution mismatch | Apply 7B 40-pair adapter, eval on BCB-Hard | No transfer |

---

## Notes on stochasticity

- **vLLM sampling** is deterministic given a fixed seed, but vLLM 0.8.x occasionally changes pad/EOS handling between point releases. Pin to 0.8.5.
- **LoRA training is seed-sensitive at small N.** The 7B 40-pair run spans 95–112/164 across seeds 7/13/42. The 14B 100-pair run is much tighter (130–134/164).
- **Stop tokens matter.** Use `--stop "\nclass " --stop "\nif __name__"` for raw-completion eval. Wrong stop tokens cut output prematurely and produce artifactually low baselines. We saw this earlier in the project — see paper §2.

---

## Cost reference (May 2026, RunPod)

| Workflow | Hardware | Wall time | Cost |
|----------|----------|-----------|------|
| 7B headline (fast path) | RTX 6000 Ada 48GB | ~30 min | ~$0.50 |
| 14B 80.5% (fast path) | H100 80GB | ~30 min | ~$1.50 |
| 14B 80.5% full path (mining + train) | H100 80GB | ~95 min | ~$3.50 |
| GSM8K 32→66 | RTX 6000 Ada | ~30 min | ~$0.50 |
| Full eval matrix (9 models) | H100 80GB | ~3 hrs | ~$8 |

Total cost to verify all numbers in the paper via the fast path: **under $10**.
