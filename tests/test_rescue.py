"""Behavior of the rescue pipeline a caller can observe."""

from pathlib import Path

import pytest
from Bio.Align.substitution_matrices import load as load_matrix

from peep_tide import (
    AminoAcid,
    AggregationPropensity,
    ConfidentRescue,
    ConservativeBound,
    Descriptor,
    EpistemicUncertainty,
    Grounding,
    InputError,
    Mutation,
    NoConfidentRescue,
    RankedCandidate,
    RescueConfig,
    choose_rescue,
    rescue,
    train,
)

FIXTURE = Path(__file__).parent / "fixtures" / "train.fasta"
MATRIX = load_matrix("BLOSUM62")
LETTERS = "ACDEFGHIKLMNPQRSTVWY"


class Scripted:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def generate(self, prompt: str) -> str:
        return self.replies.pop(0)


def _candidate(sequence, mutations, propensity):
    return RankedCandidate(
        sequence,
        mutations,
        AggregationPropensity(propensity),
        EpistemicUncertainty(0.0),
        ConservativeBound(propensity),
    )


def _expected_removed(wild_type: str, threshold: int) -> tuple[int, bool]:
    removed = 0
    proline = False
    letters = list(wild_type)
    groups = []
    for position, original in enumerate(letters):
        for letter in LETTERS:
            if letter != original:
                groups.append(((position, letter),))
    for left in range(6):
        for right in range(left + 1, 6):
            for first in LETTERS:
                if first == letters[left]:
                    continue
                for second in LETTERS:
                    if second == letters[right]:
                        continue
                    groups.append(((left, first), (right, second)))
    for mutations in groups:
        if any(float(MATRIX[wild_type[pos], letter]) < threshold for pos, letter in mutations):
            removed += 1
            if any(letter == "P" for _pos, letter in mutations):
                proline = True
    return removed, proline


@pytest.fixture(scope="module")
def oracle():
    return train(FIXTURE)


def test_rejects_a_wild_type_that_is_not_six_standard_residues(oracle):
    with pytest.raises(InputError, match="VVVVV"):
        rescue("VVVVV", RescueConfig(), oracle, Scripted(["ok"]))
    with pytest.raises(InputError, match="VVVVVX"):
        rescue("VVVVVX", RescueConfig(), oracle, Scripted(["ok"]))


def test_same_input_returns_the_same_report(oracle):
    first = rescue("EEEEEE", RescueConfig(beta=0), oracle, Scripted(["no numbers here"]))
    second = rescue("EEEEEE", RescueConfig(beta=0), oracle, Scripted(["no numbers here"]))
    assert first.wild_type == "EEEEEE"
    assert first.wild_type_propensity.value == second.wild_type_propensity.value
    assert [row.bound.value for row in first.conservative] == [row.bound.value for row in second.conservative]


def test_threshold_changes_survival_and_the_removed_count_matches_blosum(oracle):
    loose = rescue("EEEEEE", RescueConfig(blosum_threshold=-10, beta=0), oracle, Scripted(["ok"]))
    tight = rescue("EEEEEE", RescueConfig(blosum_threshold=50, beta=0), oracle, Scripted(["ok"]))
    assert loose.filter.removed_count != tight.filter.removed_count
    assert (tight.filter.removed_count, tight.filter.proline_substitution_removed) == _expected_removed("EEEEEE", 50)
    assert (loose.filter.removed_count, loose.filter.proline_substitution_removed) == _expected_removed("EEEEEE", -10)
    for report in (loose, tight):
        for row in report.conservative:
            assert len(row.mutations) in (1, 2)
            assert row.uncertainty.value >= 0


def test_black_box_orders_by_prediction_and_conservative_by_count_then_bound(oracle):
    report = rescue("EEEEEE", RescueConfig(beta=0), oracle, Scripted(["ok"]))
    propensities = [row.propensity.value for row in report.black_box]
    assert propensities == sorted(propensities)
    keys = [(len(row.mutations), row.bound.value) for row in report.conservative]
    assert keys == sorted(keys)


def test_a_bound_at_or_above_half_is_not_a_rescue():
    high = _candidate("AAAAAA", (Mutation(0, AminoAcid.CYS),), 0.5)
    low = _candidate("CCCCCC", (Mutation(0, AminoAcid.ALA), Mutation(1, AminoAcid.ASP)), 0.1)
    assert choose_rescue([high, low]).sequence == "CCCCCC"
    assert choose_rescue([high]) is None


def test_fewer_mutations_win_then_the_lower_bound_then_the_sequence():
    one_worse = _candidate("MMMMMM", (Mutation(0, AminoAcid.ALA),), 0.4)
    two_better = _candidate("NNNNNN", (Mutation(0, AminoAcid.ALA), Mutation(1, AminoAcid.CYS)), 0.1)
    assert choose_rescue([two_better, one_worse]).sequence == "MMMMMM"
    closer = _candidate("CCCCCC", (Mutation(0, AminoAcid.ASP),), 0.2)
    assert choose_rescue([one_worse, closer]).sequence == "CCCCCC"
    tie_b = _candidate("BBBBBB", (Mutation(2, AminoAcid.GLY),), 0.2)
    tie_a = _candidate("AAAAAA", (Mutation(2, AminoAcid.GLY),), 0.2)
    assert choose_rescue([tie_b, tie_a]).sequence == "AAAAAA"


def test_no_survivor_is_no_confident_rescue_without_a_feature_diff(oracle):
    report = rescue("EEEEEE", RescueConfig(blosum_threshold=50), oracle, Scripted(["unused"]))
    assert isinstance(report.outcome, NoConfidentRescue)
    assert report.rescue_row() is None
    assert report.conservative == ()


def test_beta_changes_bounds(oracle):
    calm = rescue("EEEEEE", RescueConfig(beta=0), oracle, Scripted(["ok"]))
    harsh = rescue("EEEEEE", RescueConfig(beta=50), oracle, Scripted(["ok"]))
    assert any(row.uncertainty.value > 0 for row in calm.conservative)
    assert [row.bound.value for row in calm.conservative] != [row.bound.value for row in harsh.conservative]


def test_feature_diff_has_the_five_descriptors_only_for_a_rescue(oracle):
    report = rescue("EEEEEE", RescueConfig(beta=0), oracle, Scripted(["associated with the descriptors"]))
    assert isinstance(report.outcome, ConfidentRescue)
    changes = report.outcome.feature_diff.changes
    assert [change.descriptor for change in changes] == list(Descriptor)
    for change in changes:
        assert change.change == pytest.approx(change.candidate_value - change.wild_type_value)
    blocked = rescue("EEEEEE", RescueConfig(blosum_threshold=50), oracle, Scripted(["unused"]))
    assert isinstance(blocked.outcome, NoConfidentRescue)


def test_grounding_accepts_a_diff_number_and_flags_inventions(oracle):
    probe = rescue("EEEEEE", RescueConfig(beta=0), oracle, Scripted(["associated with the descriptors"]))
    assert isinstance(probe.outcome, ConfidentRescue)
    number = f"{probe.outcome.feature_diff.changes[0].change:.4f}"
    grounded = rescue(
        "EEEEEE",
        RescueConfig(beta=0),
        oracle,
        Scripted([f"This change {number} is associated with aggregation."]),
    )
    assert grounded.outcome.explanation.grounding is Grounding.VERIFIED
    invented = rescue(
        "EEEEEE",
        RescueConfig(beta=0),
        oracle,
        Scripted(["associated with 9.9999", "associated with 9.9999"]),
    )
    assert invented.outcome.explanation.grounding is Grounding.NOT_VERIFIED
    retried = rescue(
        "EEEEEE",
        RescueConfig(beta=0),
        oracle,
        Scripted(["residue tryptophan", f"This change {number} is associated with aggregation."]),
    )
    assert retried.outcome.explanation.grounding is Grounding.VERIFIED
    failed_twice = rescue(
        "EEEEEE",
        RescueConfig(beta=0),
        oracle,
        Scripted(["residue tryptophan", "residue proline"]),
    )
    assert failed_twice.outcome.explanation.grounding is Grounding.NOT_VERIFIED
    assert "proline" in failed_twice.outcome.explanation.text
