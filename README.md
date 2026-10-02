# Calibrated, Interpretable Automated Scoring with LLM Likelihoods

This repo holds code to reproduce the four figures of the AIME 2026 work-in-progress paper Calibrated, Interpretable Automated Scoring with LLM Likelihoods.
The paper itself is `aime-2026-paper.pdf`.

## How to run

You need a CUDA GPU, Python 3.11+, and R; full list of dependencies below.

```bash
git clone https://github.com/RenaissancePhilanthropy/Calibrated-LLM-scoring-AIME-2026 && cd Calibrated-LLM-scoring-AIME-2026
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ./channel-models

# Scripts 04 and 06 serve models with vLLM, whose torch and transformers
# requirements conflict with the pins above, so it gets its own environment.
python -m venv .vllm-venv
.vllm-venv/bin/pip install vllm==0.28.0

export OPENAI_API_KEY=sk-...   # used by script 05; may live in ~/.env instead
export HF_TOKEN=hf_...         # HuggingFace token for the gated Llama-3.2 repos.
                               # Used by scripts 01-04; read from the environment or
                               # from a .env file in the repo root (git-ignored), not ~/.env

PY=.venv/bin/python bash scripts/verify_baseline.sh
```

`verify_baseline.sh` checks that your setup reproduces one cell of Fig 1: it runs the
Qwen3.5-0.8B-Base channel and direct cells end to end and prints a PASS/NEAR/FAIL
verdict against the paper's numbers. A few minutes, plus the model download.

Then run the pipeline, from the package root:

```bash
export PY=.venv/bin/python              # the Python stages read PY
export PATH=$PWD/.vllm-venv/bin:$PATH   # 04 and 06 call `vllm` to serve models

bash scripts/01_channel_sweep.sh
bash scripts/02_direct_sweep.sh
bash scripts/03_direct_corrections.sh
bash scripts/04_local_judges.sh
bash scripts/05_openai_judges.sh
bash scripts/06_interpretability.sh
bash scripts/07_export.sh
bash scripts/08_figures.sh
```

You'll find the output in `figures`.

## What each script does

| Script | Produces | Needed for |
|---|---|---|
| `01_channel_sweep.sh` | `results/channel/` — channel runs, six base models | Figs 1-4 |
| `02_direct_sweep.sh` | `results/direct/` — direct runs, same six models | Figs 1-4 |
| `03_direct_corrections.sh` | `eval_per_sample_pq_redacted.csv` and `eval_per_question_null.csv` beside each direct run | Figs 1-3 |
| `04_local_judges.sh` | `results/judge/` — the 8 locally served judge cells | Figs 1-3 |
| `05_openai_judges.sh` | same directory — the 4 OpenAI judge cells | Fig 1 |
| `06_interpretability.sh` | `figures/figdata/interp_*.csv` | Fig 4 |
| `07_export.sh` | `export/fig1_data.csv`, `figures/figdata/{channel,direct,judge}_9b.csv` | Figs 1-3 |
| `08_figures.sh` | `figures/generated/*.{pdf,png}` | all four |

The six base models are Qwen3.5-{0.8B,2B,4B,9B}-Base and Llama-3.2-{1B,3B}. Scripts
04 and 06 need vLLM on `PATH`, in its own environment (see Requirements). Before
running 08, check 07's output: a `WARNING: missing panels: {...}` line means a panel
CSV was not regenerated, so 08 would plot stale or absent data for that panel.

## Requirements and runtimes

- One A100-class GPU or better. The 9B cells need the memory headroom, both for
  scoring and for serving the judge. Channel, direct, and local judge serving are
  GPU-bound; the OpenAI judge cells are not.
- Python 3.11+.
- An OpenAI API key with a bit of money behind it, and a HuggingFace API key that gives
  access to the Llama models.
- Optional variables: `VLLM_API_KEY` is the client-side key for the local judge
  server; script 04 defaults it to `local-dummy`, and it must not reach the
  `vllm serve` process (see `judge/config.toml`). `CACHE_DIR` overrides the
  `models/` directory the channel and direct stages download weights into.
- No data ships in this repo. The dataset downloads automatically from Hugging Face
  (`nkazi/SciEntsBank`, `test_ua` split, n = 540).
- R, for the figures: `install.packages(c("ggplot2", "dplyr", "tidyr", "cowplot"))`,
  run outside any Python venv.
- vLLM 0.28.0, for scripts 04 and 06. Installed into its own environment above;
  the scripts find it by running `vllm`, so it only has to be on `PATH`. If a
  server dies at startup, read `logs/vllm_<model>.log` — the reason lands there,
  not on the console. `start_server` switches off vLLM's FlashInfer sampler,
  which would otherwise JIT-compile a kernel and need `ninja` on `PATH`; the
  judge decodes greedily, so that sampler never runs on a real request. To
  restore it, install `ninja` and set `VLLM_USE_FLASHINFER_SAMPLER=1`.
- Software these instructions were last checked against: Python 3.13.12, torch
  2.11.0+cu128, vLLM 0.28.0, R 4.5.2, ggplot2 4.0.2, dplyr 1.2.0, tidyr 1.3.2,
  cowplot 1.2.0.
- Optional, for contributors: `pip install pre-commit && pre-commit install` runs
  gitleaks and a private-key check before each commit (`.pre-commit-config.yaml`).

## What to expect

The local cells (channel, direct, locally served judges) are deterministic by
configuration: temperature 0, seed 42, greedy scoring. Numbers still move a little
across machines, because fp16/bf16 log-probs depend on the GPU model, driver, and
batch shape. In particular, using the flash-linear-attention kernels for the Qwen
models changes their results by a non-trivial amount.

Note also that the OpenAI cells may drift. Those models are not pinned artifacts,
so `05` may not reproduce the paper's numbers as the provider updates them.

`figures/reference/` holds the four figures exactly as published; compare
`figures/generated/` against them.

## Data and attribution

The dataset is SciEntsBank from SemEval-2013 Task 7 (Dzikovska et al., 2013), used
under CC BY 4.0 via its Hugging Face mirror, https://huggingface.co/datasets/nkazi/SciEntsBank.
Two items (the Fig 4 examples) are reproduced verbatim in `figures/interp_figs.R` and in
`figures/reference/interp_prefix_reveal_stacked.png`. If you use the data, cite:

```bibtex
@inproceedings{dzikovska2013semeval,
  title = {{S}em{E}val-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge},
  author = {Dzikovska, Myroslava and Nielsen, Rodney and Brew, Chris and Leacock, Claudia and Giampiccolo, Danilo and Bentivogli, Luisa and Clark, Peter and Dagan, Ido and Dang, Hoa Trang},
  year = 2013,
  booktitle = {Second Joint Conference on Lexical and Computational Semantics ({SEM})},
  pages = {263--274}
}
```

## Repo layout

```
aime-2026-paper.pdf       the paper
scientsbank.py            dataset loader (used by direct and judge)
common.py                 helpers used by more than one stage
channel/                  channel runner + config
direct/                   direct runner + config, per-question content-free calibration
judge/                    judge runner + config (local vLLM and OpenAI providers)
interpretability/         prefix-reveal traces for Fig 4
export/                   collects finished runs into the figure CSVs
figures/                  paper_figs.R, interp_figs.R, palette.R (shared colours);
                          reference/ = the published figures
scripts/                  01...08 pipeline, verify_baseline.sh, lib.sh (shared helpers)
results/                  run outputs land here (git-ignored, as are logs/ and models/)
requirements.txt          pinned package versions
channel-models/           the library for computing channel and direct method log likelihoods
```
