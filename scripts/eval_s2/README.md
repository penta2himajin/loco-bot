# S2 spikes (eval only)

Python scripts to try Liquid LFM routers for gray-zone topic decisions.
Product runtime stays Rust; these are measurement harnesses.

## Setup

```bash
python3 -m venv .tmp/venv-s2
.tmp/venv-s2/bin/pip install -r scripts/eval_s2/requirements.txt
```

## Encoder Prompt-Router (first)

```bash
.tmp/venv-s2/bin/python scripts/eval_s2/run_lfm_encoder_router.py
```

## Generative LFM2.5-350M (fallback)

```bash
.tmp/venv-s2/bin/python scripts/eval_s2/run_lfm_lm.py
```

Cases: `cases_ja.json` (Continue / New / Return / Clarify).

Measured results: see `RESULTS.md` (neither LFM variant adopted for product S2 yet).
