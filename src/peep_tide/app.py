"""One Streamlit screen. It calls rescue and does not rank on its own."""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from peep_tide import (
    ConfidentRescue,
    InputError,
    NoConfidentRescue,
    RescueConfig,
    rescue,
    train,
)
from peep_tide.model_loader import load_language_model

DATA = Path("data/Hex1279.fasta")


def _rows(report_rows):
    return [
        {
            "sequence": row.sequence,
            "mutations": len(row.mutations),
            "propensity": row.propensity.value,
            "uncertainty": row.uncertainty.value,
            "bound": row.bound.value,
        }
        for row in report_rows
    ]


def render() -> None:
    st.title("Peep-tide")
    sequence = st.text_input("Sequence", value="VQIVYK")
    threshold = st.number_input("BLOSUM threshold", value=-1, step=1)
    beta = st.number_input("beta", value=1.0, min_value=0.0, step=0.1)
    if not DATA.exists():
        st.error(f"training file is missing: {DATA}")
        return
    if "oracle" not in st.session_state:
        st.session_state.oracle = train(DATA)
    if "language_model" not in st.session_state:
        st.session_state.language_model = load_language_model(RescueConfig())
    try:
        report = rescue(
            sequence,
            RescueConfig(blosum_threshold=int(threshold), beta=float(beta)),
            st.session_state.oracle,
            st.session_state.language_model,
        )
    except InputError as exc:
        st.error(str(exc))
        return
    st.write(f"Wild-type propensity {report.wild_type_propensity.value:.3f}")
    st.write(f"Wild-type uncertainty {report.wild_type_uncertainty.value:.3f}")
    st.write(f"Removed by the evolutionary validity filter: {report.filter.removed_count}")
    flag = "yes" if report.filter.proline_substitution_removed else "no"
    st.write(f"Proline substitution removed: {flag}")
    left, right = st.columns(2)
    left.subheader("Peep-tide")
    left.dataframe(_rows(report.conservative), hide_index=True)
    right.subheader("Black-box")
    right.dataframe(_rows(report.black_box), hide_index=True)
    disagreed = sorted(report.disagreements())
    shown = ", ".join(disagreed) if disagreed else "none"
    st.write(f"Disagreements: {shown}")
    if report.black_box:
        pick = report.black_box[0]
        st.write(f"Black-box pick uncertainty {pick.uncertainty.value:.3f}")
        top = report.black_box[:12]
        figure = go.Figure(go.Scatter(
            x=[row.sequence for row in top],
            y=[row.propensity.value for row in top],
            error_y={"array": [row.uncertainty.value for row in top]},
            mode="markers",
        ))
        figure.update_layout(yaxis_title="aggregation propensity", xaxis_title="sequence")
        st.plotly_chart(figure)
    if isinstance(report.outcome, NoConfidentRescue):
        st.info("no confident rescue")
        return
    if isinstance(report.outcome, ConfidentRescue):
        st.subheader("Explanation")
        st.write(report.outcome.explanation.text)
        st.write(f"Grounding: {report.outcome.explanation.grounding.value}")
        if report.outcome.explanation.grounding.value == "not_verified":
            st.warning("This explanation is not verified against the feature diff.")
        for change in report.outcome.feature_diff.changes:
            st.write(
                f"{change.descriptor.value}: {change.wild_type_value:.4f} to "
                f"{change.candidate_value:.4f} ({change.change:+.4f})"
            )


if __name__ == "__main__":
    render()
