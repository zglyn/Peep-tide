# Peep-tide

Uncertainty-aware peptide design. See `GLOSSARY.md` for terms and `docs/adr/` for decisions.

## Setup

Python 3.12 via `uv`. From the repo root:

    uv sync

`pyproject.toml` is the dependency list. `uv.lock` pins the resolve. Torch is taken from the CUDA 12.8 index declared in `pyproject.toml` (`https://download.pytorch.org/whl/cu128`). If that index cannot be selected, install torch first and then sync the rest:

    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv sync

`requirements.lock` is the earlier pip freeze, kept so the CUDA build that was installed on this machine stays recorded.

## Commands

From the repo root, with `data/Hex1279.fasta` and `data/Hex142.fasta` present for the two reports:

    uv run pytest
    uv run streamlit run src/peep_tide/app.py
    uv run python -m peep_tide.calibration
    uv run python -m peep_tide.validity

`pytest` uses a fixture and a fake language model. The real-model smoke test is marked `slow` and stays out of that default run.

## Commands

From the repo root, with `data/Hex1279.fasta` and `data/Hex142.fasta` present for the two reports:

    uv run pytest
    uv run streamlit run src/peep_tide/app.py
    uv run python -m peep_tide.calibration
    uv run python -m peep_tide.validity

`pytest` uses a fixture and a fake language model. The real-model smoke test is marked `slow` and stays out of that default run.

## Data (not in git)

`data/Hex1279.fasta` (train) and `data/Hex142.fasta` (test) come from the
[ANuPP datasets page](https://web.iitm.ac.in/bioinfo2/ANuPP/datasets/). They are compiled from
CPAD 2.0 and WALTZ-DB 2.0, so they are not pure WALTZ-DB. The page states no license. The
original WALTZ-DB site (waltzdb.switchlab.org) was down on 2026-10-21.

## Model

`Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit (NF4). Uses about 5.9 GB of VRAM on the RTX 5070 Laptop.
