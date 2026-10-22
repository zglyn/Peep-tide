# Peep-tide

Uncertainty-aware peptide design. See `GLOSSARY.md` for terms and `docs/adr/` for decisions.

## Setup

Python 3.12 via `uv`. PyTorch needs the CUDA 12.8 build:

    uv venv --python 3.12
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install numpy pandas scikit-learn scipy gpytorch biopython modlamp streamlit plotly transformers accelerate bitsandbytes pytest

`requirements.lock` records the exact versions installed.

## Data (not in git)

`data/Hex1279.fasta` (train) and `data/Hex142.fasta` (test) come from the
[ANuPP datasets page](https://web.iitm.ac.in/bioinfo2/ANuPP/datasets/). They are compiled from
CPAD 2.0 and WALTZ-DB 2.0, so they are not pure WALTZ-DB. The page states no license. The
original WALTZ-DB site (waltzdb.switchlab.org) was down on 2026-10-21.

## Model

`Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit (NF4). Uses about 5.9 GB of VRAM on the RTX 5070 Laptop.
