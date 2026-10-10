"""Thin public name; parameter creation and computation stay in frozen source."""
from __future__ import annotations

from typing import Any

from my_experiments.models.dti_models import DTIModel


class SAGFDTI(DTIModel):
    """Original Gaussian node fusion with the legacy checkpoint state keys.

    ``config`` accepts the archived trainer configuration, including its nested
    ``model`` dictionary. The experiment is always ``E4_GAUSS_NODE_FUSION``.
    Training recipe validation belongs to the portable command-line wrappers.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("E4_GAUSS_NODE_FUSION", config)
