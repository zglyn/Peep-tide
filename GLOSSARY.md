# Peep-tide

Uncertainty-aware peptide design: given a peptide prone to aggregation, find the fewest, evolutionarily plausible point mutations that confidently reduce that tendency, and explain why.

## Language

**Peep-tide**:
The product and the method. The single name for both the app and the technique.
_Avoid_: PEP-CLUE

**Wild-type**:
The user-supplied peptide sequence to be rescued. For now, exactly six residues long.
_Avoid_: Original, input peptide, baseline sequence

**Aggregation propensity**:
The probability that a peptide forms amyloid-like aggregates, as labelled in experimental data. The single target property Peep-tide predicts.
_Avoid_: Solubility, amyloidogenicity, risk (unqualified)

**Mutation**:
A single-residue substitution in the wild-type.
_Avoid_: Edit, change, variant

**Candidate**:
A mutant of the wild-type with one or two mutations that survives the evolutionary validity filter.
_Avoid_: Counterfactual (unless meaning the chosen outcome), proposal

**Evolutionary validity filter**:
The constraint that discards any candidate containing a mutation scoring below a configurable BLOSUM62 threshold.
_Avoid_: BLOSUM filter, similarity cutoff

**Epistemic uncertainty**:
The model's uncertainty about a candidate's aggregation propensity due to lack of similar training data, as opposed to noise in the labels.
_Avoid_: Confidence (inverted), error bar

**Conservative bound**:
A candidate's predicted aggregation propensity plus uncertainty-weighted slack. Lower is better; an uncertain candidate is penalized.
_Avoid_: UCB score, risk score

**Black-box ranking**:
Ranking candidates by predicted aggregation propensity alone, ignoring epistemic uncertainty. The baseline Peep-tide is compared against.
_Avoid_: Deterministic baseline, point-estimate ranking

**Rescue**:
A candidate whose conservative bound falls below 0.5. Among rescues, fewer mutations win first, then the lower bound.
_Avoid_: Fix, success, hit

**No confident rescue**:
The valid outcome when no candidate qualifies as a rescue. Shown to the user, never hidden.
_Avoid_: Failure, empty result

**Descriptor**:
A physicochemical property of a sequence: hydrophobicity, net charge at pH 7, β-sheet propensity, Boman index, or aliphatic index.
_Avoid_: Feature (when meaning a sequence property), attribute

**Feature diff**:
The change in each descriptor between the wild-type and a candidate. The only evidence the explanation may cite.
_Avoid_: Causal diff, causal breakdown

**Explanation**:
Natural-language account of why a rescue works, grounded strictly in the feature diff.
_Avoid_: Reasoning, justification

**Module**:
One of four in-process components (input, perturbation, oracle, reasoner) behind plain function interfaces.
_Avoid_: Microservice, service
