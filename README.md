# TinyForge-Zero

**Self-bootstrapping recipes for open base LLMs — no human-written training data.**

A 14B open base model reaches **80% on HumanEval** and **74.4% on HumanEval+** with only a Python interpreter as oracle and no human-curated training data, for under **$5** of consumer-GPU compute. This repo contains the recipes, mined pairs, evaluation scripts, and adapters from the paper.

📄 **Paper**: *How Far Can an Open Base Model Self-Improve? Recipes, Limits, and Test-Time Synergy* — arXiv link forthcoming
📦 **Companion to**: `ranausmanai/tinyforge` (earlier exploratory experiments)

---

![Recipe lift vs base capability — recipe captures headroom, saturates near ceiling](docs/scaling_chart.png)

## Headline results

| Model | Setting | Base | After recipe | Δ |
|-------|---------|-----:|-------------:|--:|
| Qwen2.5-14B-Base | HumanEval (chat-template) | 44/164 (26.8%) | **131/164 (79.9%)** | **+53.0pp** |
| Qwen2.5-14B-Base | HumanEval+ | — | **122/164 (74.4%)** | — |
| Qwen2.5-7B-Base | HumanEval (best seed) | 25/164 (15.2%) | **112/164 (68.3%)** | **+53.0pp** |
| Qwen2.5-3B-Base | GSM8K (auto-difficulty curriculum) | 32/100 | **66/100** | **+34pp** |
| Random external pairs | HumanEval (control) | 25 | 25 | **+0** |

All numbers from `result.json` files in this repo's accompanying paper data. Same adapter under the multi-pair run's eval format reads **132/164 (80.5%)** — both round to 80%.

---

## The recipe in one diagram

![The TinyForge-Zero recipe — 5 stages from problem generation to evaluation](docs/recipe_diagram.png)

A control experiment — replacing the mined pairs with **identically-formatted but randomly-corrupted external pairs** — yields **exactly +0**. The signal is in the self-mined content, not the training-data format.

---

## What's in this repo

```
tinyforge-zero/
├── recipe/                                  # Training pipelines
│   ├── train_on_pairs.py                    # Fast-path: train LoRA on a released pairs.jsonl
│   ├── bootstrap.py                         # Self-bootstrap pipeline (mining + train, 7B / 3B)
│   ├── bootstrap_14b_4bit_harvest.py        # 4-bit harvest variant (when full-precision OOMs)
│   ├── multi_pair_14b.py                    # Aggressive multi-pair variant → 80.5% on 14B
│   ├── curriculum_math.py                   # Auto-difficulty curriculum for GSM8K (§2.3, §3.8)
│   ├── curriculum_code.py                   # Auto-difficulty curriculum for code
│   └── math_bootstrap.py                    # Vanilla math bootstrap (regressed; see §3.8)
├── evals/                                   # Evaluation harnesses
│   ├── eval_raw.py                          # HumanEval / MBPP / GSM8K (vLLM, raw-completion)
│   ├── eval_plus.py                         # HumanEval+ contamination-resistant eval
│   └── confirm.py                           # Confirmation re-eval against base
├── tts/                                     # Test-time sampling (§2.2, §3.3)
│   ├── tts_scaling.py                       # Pass@N scaling sweep (HE, HE+, MATH-500)
│   ├── tts_humaneval.py                     # Best-of-N pass@1 on HE/HE+
│   ├── tts_math500.py                       # Best-of-N pass@1 on MATH-500
│   ├── tts_aime.py                          # Pass@k curve on AIME (k=1..64)
│   ├── tts_qwen14b_recipe.py                # TTS on top of the 14B multi-pair adapter
│   └── tts_qwen3_8b_raw_control.py          # Control: TTS on raw Qwen3-8B (recipe vs sampling)
├── experiments/                             # Every paper experiment, one script each
│   ├── self_consistency.py                  # §3.4 — deployable TTS via majority vote (no oracle)
│   ├── recipe_x_tts_synergy.py              # §3.5 — recipe × TTS synergy threshold (novel finding)
│   ├── cross_domain_code_to_math.py         # §3.10 — code-trained recipe on math (+2, marginal)
│   ├── mbpp_seeded_cross_arch.py            # §3.9 — Llama/Coder cross-architecture self-mining
│   ├── diversity_cued_mining.py             # §3.10 — diversity-cued mining (low yield)
│   ├── recursive_bootstrap.py               # §3.10 — recursive iter1→iter2→iter3 (plateau)
│   ├── self_correction_code.py              # §3.10 — code self-correction recipe
│   ├── self_correction_math_naive.py        # §3.10 — naive (wrong→fix only): catastrophic regress
│   ├── self_correction_math_fixed.py        # §3.10 — fixed (mixed positives): recovered
│   ├── math500_seeded_mining.py             # §3.10 — distribution-mismatch demo (catastrophic)
│   ├── aime_scaling.py                      # AIME pass@k = 1..64 sweep
│   ├── bcb_hard_eval.py                     # §3.10 — BigCodeBench-Hard distribution mismatch
│   └── star_baseline_gsm8k.py               # Related-work baseline (STaR / rejection sampling FT)
├── controls/
│   └── mbpp_corrupt_control.py              # §3.6 — the +0 negative-control experiment
├── data/                                    # Released mined pairs (drove paper numbers)
│   ├── pairs_7b_40.jsonl                    # 40 pairs for Qwen2.5-7B-Base
│   ├── pairs_14b_multi_new60.jsonl          # 60 aggressive-mined pairs for 14B (+ warmup 40 = 100)
│   └── pairs_math_13.jsonl                  # 13 curriculum-mined math pairs (3B GSM8K)
├── docs/
│   ├── recipe_diagram.png                   # The 5-stage recipe diagram (rendered above)
│   ├── scaling_chart.png                    # Recipe lift vs base capability (paper Fig 1)
│   ├── fig1_headline.png                    # Headline result chart
│   └── fig6_boundary.png                    # Boundary conditions across 9 models
├── scripts/
│   └── make_recipe_diagram.py               # Source for the rendered recipe diagram
├── REPRODUCE.md                             # Paper claim → exact command mapping (all sections)
├── requirements.txt
└── LICENSE
```

A note on these scripts: `recipe/`, `evals/`, and `controls/` are the clean replication paths — these have argparse CLIs and produce the headline numbers. The scripts under `experiments/` and `tts/` are the **original research scripts** used to produce each figure / table in the paper. They work, but they're closer to "research code" than "production tooling" — argument names vary, some have hard-coded paths to `/workspace/`, and they were each run on RunPod with a specific GPU. Read the top-of-file docstring of any experiment script for what it does and how to invoke it.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/ranausmanai/tinyforge-zero.git
cd tinyforge-zero

# 2. Install (Python 3.10+, CUDA 12.1+, GPU with ≥40GB VRAM recommended)
pip install -r requirements.txt

# 3. Baseline the model (so you know the lift is real)
python evals/eval_raw.py \
    --model Qwen/Qwen2.5-7B \
    --bench humaneval

# 4. Train on the released 40 mined pairs (~10 min on H100)
python recipe/train_on_pairs.py \
    --model Qwen/Qwen2.5-7B \
    --pairs data/pairs_7b_40.jsonl \
    --epochs 2 --lr 1e-4 --lora-rank 16 \
    --out adapter_7b --seed 13

# 5. Evaluate the trained adapter
python evals/eval_raw.py \
    --model Qwen/Qwen2.5-7B \
    --adapter adapter_7b \
    --bench humaneval
```

Expected outcome: HumanEval moves from ~25/164 to **~95–112/164** (seed-dependent).

For the **14B → 80.5%** run, use `recipe/multi_pair_14b.py` with both `data/pairs_7b_40.jsonl` (warmup) and `data/pairs_14b_multi_new60.jsonl`. See [REPRODUCE.md](REPRODUCE.md) for the exact command and expected hardware.

---

## Boundary conditions (where the recipe fails)

![Recipe boundary conditions across 9 base models](docs/fig6_boundary.png)

The recipe works under stated conditions. We document four failure modes:

1. **Saturation**: Qwen3-8B/14B-Base and Qwen2.5-72B-Base have so little headroom on HumanEval that mining produces zero or negative lift.
2. **Distribution mismatch**: Pairs mined on simple problems do not transfer to BigCodeBench-Hard (library code) or MATH-500 (competition math). Catastrophic when ignored — see the over-correction case (Qwen3-4B MATH-500 dropped 299 → 69).
3. **Base capability floor**: OLMo-2-7B at 5/164 baseline produces too few "fix" attempts to mine from.
4. **Self-correction trained on wrong→fix only**: model over-doubts and degrades on correct outputs. Mixing right→stays-right traces recovers it.

See the paper's §3 for measurements; the boundary chart above shows the recipe's lift across all 9 base models we tested.

---

## Adapters

The LoRA adapter weights for the headline 14B run (the 80.5% adapter) are ~200 MB and are not committed to this repo. They live separately:

- **Hugging Face Hub**: [`ranausmans/tinyforge-zero-qwen25-14b-lora`](https://huggingface.co/ranausmans/tinyforge-zero-qwen25-14b-lora) — 192 MB, Apache-2.0 (inherits from Qwen2.5-14B base)

The adapter is a standard `peft` LoRA over `Qwen/Qwen2.5-14B`. Load with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "ranausmans/tinyforge-zero-qwen25-14b-lora")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
```

---

## Hardware used in the paper

| Run | GPU | Time | Cost |
|-----|-----|------|------|
| Qwen2.5-7B 40-pair recipe | RTX 6000 Ada | ~30 min | <$1 |
| Qwen2.5-14B multi-pair (80.5%) | 1× H100 80GB | ~95 min | ~$3.50 |
| Qwen2.5-3B GSM8K curriculum | RTX 6000 Ada | ~30 min | <$1 |
| Full eval suite (9 models, HE+HE++MBPP) | 1× H100 | ~3 hrs | ~$8 |

All runs were on rented consumer/cloud GPUs (RunPod). Total spend documented in the paper was under $50.

---

## Citation

```bibtex
@misc{usman2026tinyforgezero,
  title  = {How Far Can an Open Base Model Self-Improve?
            Recipes, Limits, and Test-Time Synergy},
  author = {Rana Usman},
  year   = {2026},
  eprint = {TBD},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI}
}
```

---

## License

MIT — see [LICENSE](LICENSE). The mined pairs in `data/` are derivatives of base-model outputs (Qwen2.5 family, Apache-2.0). Treat downstream redistribution accordingly.

---

## Contact

- Issues / questions: [GitHub Issues](https://github.com/ranausmanai/tinyforge-zero/issues)
- Email: usmanashrafrana@gmail.com
