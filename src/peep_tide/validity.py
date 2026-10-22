"""Disruptor-residue rates for both rankings.

An introduction is a mutation to Pro, Gly, Asp, Glu, Lys, or Arg.
The rate is the share of recommendations that contain at least one.
"""

from pathlib import Path

from peep_tide import AminoAcid, ConfidentRescue, RescueConfig, load_peptides, rescue, train
from peep_tide.model_loader import load_language_model

TRAIN = Path("data/Hex1279.fasta")
HELD_OUT = Path("data/Hex142.fasta")
DISRUPTORS = {
    AminoAcid.PRO,
    AminoAcid.GLY,
    AminoAcid.ASP,
    AminoAcid.GLU,
    AminoAcid.LYS,
    AminoAcid.ARG,
}


def _introduces(mutations) -> bool:
    return any(mutation.introduced in DISRUPTORS for mutation in mutations)


def main(training: Path = TRAIN, held_out: Path = HELD_OUT, config: RescueConfig | None = None) -> None:
    chosen = config or RescueConfig()
    oracle = train(training)
    language_model = load_language_model(chosen)
    no_confident = 0
    black_hits = 0
    black_total = 0
    rescue_hits = 0
    rescue_total = 0
    for sequence in load_peptides(held_out):
        report = rescue(sequence, chosen, oracle, language_model)
        if report.black_box:
            black_total += 1
            if _introduces(report.black_box[0].mutations):
                black_hits += 1
        if isinstance(report.outcome, ConfidentRescue):
            rescue_total += 1
            if _introduces(report.outcome.candidate.mutations):
                rescue_hits += 1
        else:
            no_confident += 1
    black_rate = black_hits / black_total if black_total else float("nan")
    rescue_rate = rescue_hits / rescue_total if rescue_total else float("nan")
    print(f"black-box disruptor rate {black_rate}")
    print(f"peep-tide disruptor rate {rescue_rate}")
    print(f"no confident rescue {no_confident}")


if __name__ == "__main__":
    main()
