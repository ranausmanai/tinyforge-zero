# Reproduction Guide

Maps every paper claim → the script that produced it. Two replication paths:

- **Fast path** — use `recipe/train_on_pairs.py` with `data/*.jsonl`. Reproduces the trained adapter and headline number in ~30 min on H100. Recommended for paper verification.
- **Full path** — re-run the original research scripts end-to-end including the self-mining stage. Use this if applying the recipe to a *new* base model.

A note on script conventions: scripts under `recipe/`, `evals/`, and `controls/` are clean replication paths (argparse CLIs, no hardcoded paths). Scripts under `experiments/` and `tts/` are the original research code used to produce each finding — they work but use `--tag`-style outputs and sometimes assume `/workspace/` paths (set via `HF_HOME` env var). Read the top-of-file docstring of each to see exact invocation.

---

## Environment

Tested on:
- **H100 80GB** — Debian 12, CUDA 12.4, driver 570+ (required for vLLM 0.8.5)
- **RTX 6000 Ada 48GB** — sufficient for ≤7B models

```bash
pip install -r requirements.txt
```

Pinned stack: `torch==2.6.0`, `transformers==4.51.3`, `vllm==0.8.5`, `peft==0.13.0`.

---

# Mapping: paper claim → script

## §2 Method

| Paper § | Method | Script | Notes |
|---|---|---|---|
| §2.1 | Self-bootstrap pipeline (code) | `recipe/bootstrap.py` | Generation → solving → mining → train, end-to-end |
| §2.1 | 4-bit harvest for large models | `recipe/bootstrap_14b_4bit_harvest.py` | NF4 quantization, harvest-only (no in-loop training) |
| §2.1 | Aggressive multi-pair mining | `recipe/multi_pair_14b.py` | The 14B 80.5% pipeline |
| §2.2 | Test-time sampling (oracle) | `tts/tts_scaling.py` | Pass@N for HE / HE+ / MATH-500 |
| §2.3 | Auto-difficulty curriculum (math) | `recipe/curriculum_math.py` | The GSM8K 32→66 pipeline |
| §2.3 | Auto-difficulty curriculum (code) | `recipe/curriculum_code.py` | Code variant |

---

## §3 Experiments

### §3.2 Recipe alone — HumanEval and HumanEval+

| Claim (paper Table 1) | Script + command |
|---|---|
| Qwen2.5-7B-Base: 25 → 112 (+87 best seed) | Fast path: `python recipe/train_on_pairs.py --model Qwen/Qwen2.5-7B --pairs data/pairs_7b_40.jsonl --seed 13 --lora-rank 16 --out adapter_7b_seed13` then `python evals/eval_raw.py --model Qwen/Qwen2.5-7B --adapter adapter_7b_seed13 --bench humaneval` |
| Qwen2.5-14B-Base: 44 → 131 / 80% on HE, 122/164 on HE+ | `cat data/pairs_7b_40.jsonl data/pairs_14b_multi_new60.jsonl > /tmp/14b.jsonl; python recipe/train_on_pairs.py --model Qwen/Qwen2.5-14B --pairs /tmp/14b.jsonl --lora-rank 32 --out adapter_14b_multi; python evals/eval_plus.py --model Qwen/Qwen2.5-14B --adapter adapter_14b_multi` |
| Multi-pair full path (re-mine + train) | `python recipe/multi_pair_14b.py --model Qwen/Qwen2.5-14B --warmup_pairs_path data/pairs_7b_40.jsonl --n_problems 200 --n_attempts 8 --max_pairs_per_problem 4 --lora_rank 32 --tag multi_rerun` |
| Boundary table for all 9 models | `python evals/eval_raw.py --model <each>` for baseline; recipe + re-eval per model. Cost: ~3 hr H100. |

### §3.3 Test-time sampling (TTS) alone

| Claim | Script | Expected |
|---|---|---|
| Qwen3-4B best-of-8 HE oracle = 92.7% | `python tts/tts_humaneval.py --model Qwen/Qwen3-4B-Base --n 8 --temperature 0.7` | 152/164 |
| Qwen3-8B best-of-8 HE oracle = 92.1% | `python tts/tts_humaneval.py --model Qwen/Qwen3-8B-Base --n 8 --temperature 0.7` | 151/164 |
| Qwen3-4B best-of-8 MATH-500 = 79.4% | `python tts/tts_math500.py --model Qwen/Qwen3-4B-Base --n 8` | 397/500 |
| Qwen3-8B best-of-8 MATH-500 = 81.0% | `python tts/tts_math500.py --model Qwen/Qwen3-8B-Base --n 8` | 405/500 |
| AIME pass@k curve (k=1..64) | `python tts/tts_aime.py --model Qwen/Qwen3-8B-Base --n 32` | 25.6 / 38.9% best-of-32 |
| Full TTS scaling sweep (Table 2) | `python tts/tts_scaling.py --model Qwen/Qwen3-4B-Base` |  |

### §3.4 Self-consistency (deployable TTS, no oracle)

```bash
python experiments/self_consistency.py \
    --model Qwen/Qwen3-4B-Base \
    --bench gsm8k --n 8
```
Tests if majority-vote selection without oracle access matches oracle pass@N. See paper Table 3.

### §3.5 Recipe × TTS synergy threshold (novel finding)

```bash
python experiments/recipe_x_tts_synergy.py \
    --base-model Qwen/Qwen2.5-14B \
    --adapter adapter_14b_multi \
    --n 8
```
Compares: raw base | raw base + TTS | recipe-trained | recipe-trained + TTS. The novel finding: at sufficient mined-pair counts, recipe-trained + TTS > raw + TTS (+12.8pp). At too-few pairs, recipe-trained + TTS < raw + TTS (-4.9pp on Qwen2.5-3B with 36 pairs).

### §3.6 Control: format alone does not explain the lift

```bash
python controls/mbpp_corrupt_control.py \
    --model Qwen/Qwen2.5-7B \
    --tag mbpp_corrupt_control
```
Expected: HumanEval stays at 25/164 (Δ = 0). Confirms the signal is in self-mined content, not pair-formatted training data.

### §3.7 Multi-pair mining at 14B (the 80.5% headline)

```bash
python recipe/multi_pair_14b.py \
    --model Qwen/Qwen2.5-14B \
    --warmup_pairs_path data/pairs_7b_40.jsonl \
    --n_problems 200 --n_attempts 8 \
    --max_pairs_per_problem 4 --lora_rank 32 \
    --tag multi_rerun
```
Expected: base 67/164 → trained 132/164 (multi-pair eval format) / 131/164 chat-template / 122/164 HE+.

### §3.8 Math: auto-difficulty curriculum

```bash
python recipe/curriculum_math.py \
    --model Qwen/Qwen2.5-3B \
    --iterations 16 \
    --tag curr_3b_rerun
```
Expected: GSM8K 32/100 → 66/100. Compare to `recipe/math_bootstrap.py` (vanilla, no curriculum) which regresses.

### §3.9 Cross-architecture and cross-generation

| Model | Script | Expected |
|---|---|---|
| Llama-3.2-3B (own-mined 32) | `python experiments/mbpp_seeded_cross_arch.py --model meta-llama/Llama-3.2-3B` | HE 39→43 (+4) |
| Qwen2.5-Coder-7B-Base | `python experiments/mbpp_seeded_cross_arch.py --model Qwen/Qwen2.5-Coder-7B` | HE 83→87 (+4), MBPP 122→124 (+2) |
| Qwen3-4B-Base | Same script, Qwen3-4B-Base | HE 79→106 (+27), MBPP 135→148 (+13) |

### §3.10 Failure modes and negative results

Each negative finding has its own script. Run any of these to verify the documented failure.

| Failure mode | Script | Expected |
|---|---|---|
| Saturation (Qwen3-8B/14B HE) | `python recipe/bootstrap.py --model Qwen/Qwen3-8B-Base --tag sat_check` | 132 → 118–133, no clean lift |
| BCB-Hard distribution mismatch | `python experiments/bcb_hard_eval.py --model Qwen/Qwen3-8B-Base --adapter adapter_7b_seed13` | No transfer; HE-style pairs don't generalize to library code |
| MATH-500 mining distribution mismatch | `python experiments/math500_seeded_mining.py --model Qwen/Qwen3-8B-Base` | 279/500 → 239/500 (−40, catastrophic) |
| Self-correction over-correction (naive) | `python experiments/self_correction_math_naive.py --model Qwen/Qwen3-4B-Base` | 299/500 → 69/500 (Δ=−230!) |
| Self-correction recovery (fixed) | `python experiments/self_correction_math_fixed.py --model Qwen/Qwen3-4B-Base` | Recovers to baseline + small lift via mixed positives |
| Recursive bootstrap plateau | `python experiments/recursive_bootstrap.py --model Qwen/Qwen2.5-7B --iters 3` | iter1 gives most lift, iter2/3 plateau |
| Cross-domain transfer (code→math) | `python experiments/cross_domain_code_to_math.py --code-adapter adapter_7b_seed13` | +2 marginal lift on GSM8K |
| Diversity-cued mining low yield | `python experiments/diversity_cued_mining.py --model Qwen/Qwen2.5-7B` | Fewer well-formed pairs than vanilla mining |

---

## §3.11 Boundary conditions summary (Figure 6)

The 9-model boundary chart is the synthesis of per-model recipe runs. To regenerate:

```bash
for MODEL in Qwen/Qwen2.5-{3B,7B,14B,72B} Qwen/Qwen3-{1.7B,4B,8B,14B}-Base meta-llama/Llama-3.2-3B Qwen/Qwen2.5-Coder-7B allenai/OLMo-2-1124-7B; do
    python evals/eval_raw.py --model "$MODEL" --bench humaneval  # baseline
    python recipe/bootstrap.py --model "$MODEL" --tag "boundary_$(echo $MODEL | tr '/' '_')"
done
```
Run time: ~3 hours on a single H100, ~$8 cost.

---

## Pair-count sweep (Figure 3)

```bash
for N in 10 21 40; do
    head -n $N data/pairs_7b_40.jsonl > /tmp/pairs_$N.jsonl
    python recipe/train_on_pairs.py \
        --model Qwen/Qwen2.5-7B \
        --pairs /tmp/pairs_$N.jsonl \
        --out adapter_n$N --epochs 2
    python evals/eval_raw.py \
        --model Qwen/Qwen2.5-7B --adapter adapter_n$N --bench humaneval
done
```
Expected: n=10 → ~51, n=21 → mean ~91, n=40 → mean ~105 (seed-dependent for small N).

---

## Related-work baseline

| Method | Script | Use |
|---|---|---|
| STaR / rejection-sampling FT on GSM8K | `experiments/star_baseline_gsm8k.py` | Comparison point for the curriculum result |

---

## Notes on stochasticity and reproducibility

- **vLLM sampling** is deterministic given a fixed seed, but vLLM 0.8.x can change pad/EOS handling between point releases. Pin to 0.8.5.
- **LoRA training is seed-sensitive at small N.** 7B 40-pair: 95–112/164 across seeds 7/13/42. 14B 100-pair: 130–134/164 (tighter).
- **Stop tokens matter.** Use `--stop "\nclass " --stop "\nif __name__"` for raw-completion eval. Wrong stop tokens cut output and produce artifactually low baselines. We hit this earlier in the project; the paper §2 documents the fix.

---

## Cost reference (May 2026, RunPod)

| Workflow | Hardware | Wall time | Cost |
|---|---|---|---|
| 7B headline (fast path) | RTX 6000 Ada 48GB | ~30 min | ~$0.50 |
| 14B 80.5% (fast path) | H100 80GB | ~30 min | ~$1.50 |
| 14B 80.5% full path | H100 80GB | ~95 min | ~$3.50 |
| GSM8K 32→66 curriculum | RTX 6000 Ada | ~30 min | ~$0.50 |
| TTS scaling sweep (one model) | H100 80GB | ~30 min | ~$1.50 |
| Full 9-model boundary chart | H100 80GB | ~3 hrs | ~$8 |
| Every negative result | mixed | ~5 hrs total | ~$15 |

Verify all paper numbers via fast path: **under $10**. Full reproduction from scratch (including all negative results and the full TTS sweep): **~$50**, matching the paper's reported total spend.
