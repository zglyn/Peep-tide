"""Peep-tide rescue pipeline.

Public entry points are train, rescue, choose_rescue, load_peptides, and
epistemic_uncertainty. Parsing, BLOSUM62, descriptor scales, and the
Gaussian process stay in this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import gpytorch
import torch
from Bio.Align.substitution_matrices import load as load_matrix
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from modlamp.descriptors import GlobalDescriptor

__all__ = [
    "train",
    "rescue",
    "choose_rescue",
    "load_peptides",
    "epistemic_uncertainty",
    "RescueConfig",
    "RescueReport",
    "RankedCandidate",
    "Mutation",
    "AminoAcid",
    "FilterReport",
    "ConfidentRescue",
    "NoConfidentRescue",
    "FeatureDiff",
    "DescriptorChange",
    "Descriptor",
    "Explanation",
    "Grounding",
    "AggregationPropensity",
    "EpistemicUncertainty",
    "ConservativeBound",
    "Oracle",
    "LanguageModel",
    "InputError",
]

__version__ = "0.1.0"

RESCUE_THRESHOLD = 0.5

# Kyte and Doolittle, J Mol Biol 1982. Higher means more hydrophobic.
_KYTE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Chou and Fasman, Biochemistry 1978, Table 1, P_beta.
_CHOU_BETA = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}
_AA_NAMES = {
    "alanine", "cysteine", "aspartic", "aspartate", "glutamic", "glutamate",
    "phenylalanine", "glycine", "histidine", "isoleucine", "lysine", "leucine",
    "methionine", "asparagine", "proline", "glutamine", "arginine", "serine",
    "threonine", "valine", "tryptophan", "tyrosine",
}
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_TOKEN = re.compile(r"[A-Za-z]+")


class InputError(ValueError):
    """A wild-type or a fasta record cannot be read."""


class AminoAcid(Enum):
    ALA = "A"
    CYS = "C"
    ASP = "D"
    GLU = "E"
    PHE = "F"
    GLY = "G"
    HIS = "H"
    ILE = "I"
    LYS = "K"
    LEU = "L"
    MET = "M"
    ASN = "N"
    PRO = "P"
    GLN = "Q"
    ARG = "R"
    SER = "S"
    THR = "T"
    VAL = "V"
    TRP = "W"
    TYR = "Y"


_BY_LETTER = {member.value: member for member in AminoAcid}


class Descriptor(Enum):
    HYDROPHOBICITY = "hydrophobicity"
    NET_CHARGE_PH7 = "net_charge_at_ph_7"
    BETA_SHEET_PROPENSITY = "beta_sheet_propensity"
    BOMAN_INDEX = "boman_index"
    ALIPHATIC_INDEX = "aliphatic_index"


@dataclass(frozen=True, slots=True)
class AggregationPropensity:
    value: float


@dataclass(frozen=True, slots=True)
class EpistemicUncertainty:
    value: float


@dataclass(frozen=True, slots=True)
class ConservativeBound:
    value: float


@dataclass(frozen=True, slots=True, kw_only=True)
class RescueConfig:
    """Keyword-only so the threshold and beta cannot be swapped by position.

    model_id is the reasoner id from issue 7. It does not change the oracle.
    """

    blosum_threshold: int = -1
    beta: float = 1.0
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    def __post_init__(self) -> None:
        if isinstance(self.blosum_threshold, bool) or not isinstance(self.blosum_threshold, int):
            raise InputError(f"BLOSUM threshold must be an integer, got {self.blosum_threshold!r}")
        if isinstance(self.beta, bool) or not isinstance(self.beta, (int, float)) or self.beta < 0:
            raise InputError(f"beta must be a non-negative number, got {self.beta!r}")
        if not self.model_id or not isinstance(self.model_id, str):
            raise InputError(f"model id must be a non-empty string, got {self.model_id!r}")


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class Mutation:
    position: int
    introduced: AminoAcid

    def __post_init__(self) -> None:
        if self.position not in (0, 1, 2, 3, 4, 5):
            raise InputError(f"mutation position must be 0 through 5, got {self.position!r}")


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    sequence: str
    mutations: tuple[Mutation, ...]
    propensity: AggregationPropensity
    uncertainty: EpistemicUncertainty
    bound: ConservativeBound

    def __post_init__(self) -> None:
        if len(self.mutations) not in (1, 2):
            raise InputError(f"a candidate has one or two mutations, got {len(self.mutations)}")
        positions = [mutation.position for mutation in self.mutations]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise InputError(f"mutation positions must increase, got {positions}")


@dataclass(frozen=True, slots=True)
class FilterReport:
    removed_count: int
    proline_substitution_removed: bool


@dataclass(frozen=True, slots=True)
class DescriptorChange:
    descriptor: Descriptor
    wild_type_value: float
    candidate_value: float

    @property
    def change(self) -> float:
        return self.candidate_value - self.wild_type_value


@dataclass(frozen=True, slots=True)
class FeatureDiff:
    changes: tuple[DescriptorChange, ...]


class Grounding(Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"


@dataclass(frozen=True, slots=True)
class Explanation:
    text: str
    grounding: Grounding


@dataclass(frozen=True, slots=True)
class ConfidentRescue:
    candidate: RankedCandidate
    feature_diff: FeatureDiff
    explanation: Explanation


@dataclass(frozen=True, slots=True)
class NoConfidentRescue:
    pass


type Outcome = ConfidentRescue | NoConfidentRescue


@dataclass(frozen=True, slots=True)
class RescueReport:
    wild_type: str
    wild_type_propensity: AggregationPropensity
    wild_type_uncertainty: EpistemicUncertainty
    filter: FilterReport
    black_box: tuple[RankedCandidate, ...]
    conservative: tuple[RankedCandidate, ...]
    outcome: Outcome

    def __post_init__(self) -> None:
        if isinstance(self.outcome, ConfidentRescue):
            bound = self.outcome.candidate.bound.value
            if bound >= RESCUE_THRESHOLD:
                raise InputError(f"confident rescue bound must be below 0.5, got {bound}")

    def disagreements(self) -> frozenset[str]:
        black = {row.sequence: index for index, row in enumerate(self.black_box)}
        conservative = {row.sequence: index for index, row in enumerate(self.conservative)}
        return frozenset(seq for seq, index in black.items() if conservative[seq] != index)

    def rescue_row(self) -> RankedCandidate | None:
        if isinstance(self.outcome, ConfidentRescue):
            return self.outcome.candidate
        return None


class _LatentGP(ApproximateGP):
    def __init__(self, inducing: torch.Tensor) -> None:
        count = inducing.shape[0]
        distribution = CholeskyVariationalDistribution(count)
        strategy = VariationalStrategy(
            self, inducing, distribution, learn_inducing_locations=False
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, inputs: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(inputs), self.covar_module(inputs)
        )


class Oracle:
    """Trained posterior. Scoring does not mutate it."""

    def __init__(
        self,
        model: _LatentGP,
        likelihood: BernoulliLikelihood,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self._model = model
        self._likelihood = likelihood
        self._center = center
        self._scale = scale

    def _score(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        standardized = (features - self._center) / self._scale
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad():
            latent = self._model(standardized)
            propensity = self._likelihood(latent).mean.clamp(0, 1)
            slope = propensity * (1 - propensity)
            uncertainty = (slope * latent.stddev).clamp_min(0)
        return propensity, uncertainty


def _mean_table(sequence: str, table: dict[str, float]) -> float:
    return sum(table[letter] for letter in sequence) / len(sequence)


def _describe(sequence: str) -> dict[Descriptor, float]:
    tool = GlobalDescriptor([sequence])
    tool.calculate_charge(ph=7.0)
    charge = float(tool.descriptor[0, 0])
    tool.boman_index()
    boman = float(tool.descriptor[0, 0])
    tool.aliphatic_index()
    aliphatic = float(tool.descriptor[0, 0])
    return {
        Descriptor.HYDROPHOBICITY: _mean_table(sequence, _KYTE),
        Descriptor.NET_CHARGE_PH7: charge,
        Descriptor.BETA_SHEET_PROPENSITY: _mean_table(sequence, _CHOU_BETA),
        Descriptor.BOMAN_INDEX: boman,
        Descriptor.ALIPHATIC_INDEX: aliphatic,
    }


def _parse_wild_type(text: str) -> str:
    if not isinstance(text, str):
        raise InputError(f"wild-type must be six standard amino acids, got {text!r}")
    stripped = text.strip()
    if any(character.isspace() for character in stripped):
        raise InputError(f"wild-type must be six standard amino acids, got {text!r}")
    sequence = stripped.upper()
    if len(sequence) != 6 or any(letter not in _BY_LETTER for letter in sequence):
        raise InputError(f"wild-type must be six standard amino acids, got {text!r}")
    return sequence


def _read_fasta(path: Path) -> tuple[tuple[str, bool], ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read fasta {path}: {exc}") from exc
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    rows: list[tuple[str, bool]] = []
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header.startswith(">") or index + 1 >= len(lines):
            raise InputError(f"malformed fasta record near {header!r}")
        sequence = lines[index + 1].upper()
        if len(sequence) != 6 or any(letter not in _BY_LETTER for letter in sequence):
            raise InputError(f"fasta sequence must be six standard amino acids, got {sequence!r}")
        if "category:amyloid" in header and "category:non-amyloid" not in header:
            amyloid = True
        elif "category:non-amyloid" in header:
            amyloid = False
        else:
            raise InputError(f"fasta header needs a category, got {header!r}")
        rows.append((sequence, amyloid))
        index += 2
    if not rows:
        raise InputError(f"fasta {path} has no records")
    return tuple(rows)


def load_peptides(path: Path) -> tuple[str, ...]:
    return tuple(sequence for sequence, _amyloid in _read_fasta(path))


def _enumerate(wild_type: str) -> tuple[tuple[str, tuple[Mutation, ...]], ...]:
    letters = list(wild_type)
    singles: list[tuple[str, tuple[Mutation, ...]]] = []
    for position, original in enumerate(letters):
        for member in AminoAcid:
            if member.value == original:
                continue
            mutated = letters.copy()
            mutated[position] = member.value
            singles.append((
                "".join(mutated),
                (Mutation(position, member),),
            ))
    doubles: list[tuple[str, tuple[Mutation, ...]]] = []
    for left in range(6):
        for right in range(left + 1, 6):
            for first in AminoAcid:
                if first.value == letters[left]:
                    continue
                for second in AminoAcid:
                    if second.value == letters[right]:
                        continue
                    mutated = letters.copy()
                    mutated[left] = first.value
                    mutated[right] = second.value
                    doubles.append((
                        "".join(mutated),
                        (Mutation(left, first), Mutation(right, second)),
                    ))
    return tuple(singles + doubles)


def _blosum_ok(mutations: Sequence[Mutation], wild_type: str, threshold: int) -> bool:
    matrix = _blosum_ok.matrix
    for mutation in mutations:
        original = wild_type[mutation.position]
        score = float(matrix[original, mutation.introduced.value])
        if score < threshold:
            return False
    return True


_blosum_ok.matrix = load_matrix("BLOSUM62")  # type: ignore[attr-defined]


def _filter(
    wild_type: str, threshold: int
) -> tuple[tuple[tuple[str, tuple[Mutation, ...]], ...], FilterReport]:
    enumerated = _enumerate(wild_type)
    kept: list[tuple[str, tuple[Mutation, ...]]] = []
    proline_removed = False
    removed = 0
    for sequence, mutations in enumerated:
        if _blosum_ok(mutations, wild_type, threshold):
            kept.append((sequence, mutations))
            continue
        removed += 1
        if any(mutation.introduced is AminoAcid.PRO for mutation in mutations):
            proline_removed = True
    return tuple(kept), FilterReport(removed, proline_removed)


def _features(sequence: str) -> list[float]:
    values = _describe(sequence)
    return [values[descriptor] for descriptor in Descriptor]


def train(training_fasta: Path) -> Oracle:
    """Fit a variational Bernoulli GP on CPU with a fixed seed.

    CPU and the fixed seed keep the same file on the same machine repeatable.
    Beta and the BLOSUM threshold are not inputs, so a second rescue reuses
    this oracle.
    """
    rows = _read_fasta(training_fasta)
    if len(rows) < 2 or len({label for _seq, label in rows}) < 2:
        raise InputError(f"training fasta {training_fasta} needs both classes")
    torch.manual_seed(0)
    features = torch.tensor([_features(sequence) for sequence, _label in rows], dtype=torch.float32)
    labels = torch.tensor([1.0 if label else 0.0 for _sequence, label in rows], dtype=torch.float32)
    center = features.mean(dim=0)
    scale = features.std(dim=0).clamp_min(1e-3)
    standardized = (features - center) / scale
    inducing_count = min(32, standardized.shape[0])
    inducing = standardized[:inducing_count].clone()
    model = _LatentGP(inducing)
    likelihood = BernoulliLikelihood()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=0.1,
    )
    objective = VariationalELBO(likelihood, model, num_data=labels.shape[0])
    for _step in range(40):
        optimizer.zero_grad()
        loss = -objective(model(standardized), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return Oracle(model, likelihood, center, scale)


def _predictions(
    oracle: Oracle, sequences: Sequence[str]
) -> tuple[tuple[float, float], ...]:
    features = torch.tensor([_features(sequence) for sequence in sequences], dtype=torch.float32)
    propensity, uncertainty = oracle._score(features)
    pairs = []
    for propensity_value, uncertainty_value in zip(propensity.tolist(), uncertainty.tolist(), strict=True):
        pairs.append((float(propensity_value), float(uncertainty_value)))
    return tuple(pairs)


def epistemic_uncertainty(sequence: str, oracle: Oracle) -> EpistemicUncertainty:
    parsed = _parse_wild_type(sequence)
    _propensity, uncertainty = _predictions(oracle, [parsed])[0]
    return EpistemicUncertainty(uncertainty)


def choose_rescue(candidates: Sequence[RankedCandidate]) -> RankedCandidate | None:
    """Fewest mutations, then lowest bound, then ascending sequence.

    A rescue is strictly below 0.5. The sequence tie-break makes the choice
    total when two rescues share a count and a bound.
    """
    eligible = [candidate for candidate in candidates if candidate.bound.value < RESCUE_THRESHOLD]
    if not eligible:
        return None
    return min(eligible, key=lambda candidate: (len(candidate.mutations), candidate.bound.value, candidate.sequence))


def _feature_diff(wild_type: str, candidate: str) -> FeatureDiff:
    left = _describe(wild_type)
    right = _describe(candidate)
    changes = tuple(
        DescriptorChange(descriptor, left[descriptor], right[descriptor])
        for descriptor in Descriptor
    )
    return FeatureDiff(changes)


def _allowed_numbers(diff: FeatureDiff) -> set[float]:
    allowed: set[float] = set()
    for change in diff.changes:
        for value in (change.wild_type_value, change.candidate_value, change.change):
            allowed.add(round(value, 4))
    return allowed


def _grounded(text: str, diff: FeatureDiff) -> bool:
    allowed = _allowed_numbers(diff)
    for match in _NUMBER.findall(text):
        if round(float(match), 4) not in allowed:
            return False
    for token in _TOKEN.findall(text):
        if len(token) == 1 and token.upper() in _BY_LETTER:
            return False
        if token.lower() in _AA_NAMES:
            return False
    return True


def _prompt(diff: FeatureDiff) -> str:
    lines = [
        "Explain the feature diff. Say associated with. Do not say cause.",
        "Use only the numbers below. Do not name amino acids.",
    ]
    directions = {
        Descriptor.HYDROPHOBICITY: "higher means more aggregation",
        Descriptor.NET_CHARGE_PH7: "higher absolute charge means less aggregation",
        Descriptor.BETA_SHEET_PROPENSITY: "higher means more aggregation",
        Descriptor.BOMAN_INDEX: "higher means less aggregation",
        Descriptor.ALIPHATIC_INDEX: "higher means more aggregation",
    }
    for change in diff.changes:
        lines.append(
            f"{change.descriptor.value} wild {change.wild_type_value:.4f} "
            f"candidate {change.candidate_value:.4f} change {change.change:.4f} "
            f"({directions[change.descriptor]})"
        )
    return "\n".join(lines)


def _explain(diff: FeatureDiff, language_model: LanguageModel) -> Explanation:
    prompt = _prompt(diff)
    text = language_model.generate(prompt)
    if _grounded(text, diff):
        return Explanation(text, Grounding.VERIFIED)
    text = language_model.generate(prompt)
    status = Grounding.VERIFIED if _grounded(text, diff) else Grounding.NOT_VERIFIED
    return Explanation(text, status)


def rescue(
    wild_type: str,
    config: RescueConfig,
    oracle: Oracle,
    language_model: LanguageModel,
) -> RescueReport:
    sequence = _parse_wild_type(wild_type)
    kept, summary = _filter(sequence, config.blosum_threshold)
    wild_propensity, wild_uncertainty = _predictions(oracle, [sequence])[0]
    scored: list[RankedCandidate] = []
    if kept:
        predictions = _predictions(oracle, [item[0] for item in kept])
        for (mutant, mutations), (propensity, uncertainty) in zip(kept, predictions, strict=True):
            bound = propensity + config.beta * uncertainty
            scored.append(RankedCandidate(
                mutant,
                mutations,
                AggregationPropensity(propensity),
                EpistemicUncertainty(uncertainty),
                ConservativeBound(bound),
            ))
    black_box = tuple(sorted(scored, key=lambda row: (row.propensity.value, row.sequence)))
    conservative = tuple(sorted(
        scored,
        key=lambda row: (len(row.mutations), row.bound.value, row.sequence),
    ))
    chosen = choose_rescue(scored)
    if chosen is None:
        outcome: Outcome = NoConfidentRescue()
    else:
        diff = _feature_diff(sequence, chosen.sequence)
        outcome = ConfidentRescue(chosen, diff, _explain(diff, language_model))
    return RescueReport(
        sequence,
        AggregationPropensity(wild_propensity),
        EpistemicUncertainty(wild_uncertainty),
        summary,
        black_box,
        conservative,
        outcome,
    )
