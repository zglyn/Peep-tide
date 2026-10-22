"""One-sided uncertainty comparison and a per-residue table.

A held-out peptide is out of distribution when its best positional identity
to any training peptide is 3 of 6 or fewer. The table is the mean uncertainty
of held-out peptides that contain each amino acid.
"""

from pathlib import Path

from scipy.stats import mannwhitneyu

from peep_tide import AminoAcid, epistemic_uncertainty, load_peptides, train

TRAIN = Path("data/Hex1279.fasta")
HELD_OUT = Path("data/Hex142.fasta")


def _identity(left: str, right: str) -> int:
    return sum(a == b for a, b in zip(left, right, strict=True))


def main(training: Path = TRAIN, held_out: Path = HELD_OUT) -> None:
    oracle = train(training)
    training_sequences = load_peptides(training)
    held_sequences = load_peptides(held_out)
    scored = []
    for sequence in held_sequences:
        uncertainty = epistemic_uncertainty(sequence, oracle).value
        best = max(_identity(sequence, other) for other in training_sequences)
        scored.append((sequence, uncertainty, best))
    out_of_distribution = [uncertainty for _seq, uncertainty, best in scored if best <= 3]
    in_distribution = [uncertainty for _seq, uncertainty, best in scored if best >= 4]
    if out_of_distribution and in_distribution:
        result = mannwhitneyu(out_of_distribution, in_distribution, alternative="greater")
        print(f"statistic {result.statistic}")
        print(f"p-value {result.pvalue}")
    else:
        print("statistic unavailable")
        print("p-value unavailable")
    for amino in AminoAcid:
        group = [uncertainty for sequence, uncertainty, _best in scored if amino.value in sequence]
        mean = sum(group) / len(group) if group else float("nan")
        print(f"{amino.value} {mean} n={len(group)}")


if __name__ == "__main__":
    main()
