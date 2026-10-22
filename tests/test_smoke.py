"""One generation with the local model. Excluded from the default pytest run."""

import pytest

from peep_tide import RescueConfig
from peep_tide.model_loader import load_language_model


@pytest.mark.slow
def test_local_model_generates_once():
    model = load_language_model(RescueConfig())
    text = model.generate("Say associated with.")
    assert text.strip()
