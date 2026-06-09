"""Model/experiment registry for reproduction scripts."""
from __future__ import annotations

from typing import Type

from recsys_edge.core import BaseExperiment
from recsys_edge.models.autorec import SpaceTrackedAutoRecExperiment
from recsys_edge.models.leaf import SpaceTrackedLeafMFExperiment
from recsys_edge.models.unisketchmf import SpaceTrackedJLRaceMFSideInfoExperiment
from recsys_edge.models.youtube import SpaceTrackedYoutubeExperiment


_EXPERIMENTS = {
    "youtube": SpaceTrackedYoutubeExperiment,
    "autorec": SpaceTrackedAutoRecExperiment,
    "leaf": SpaceTrackedLeafMFExperiment,
    "unisketchmf": SpaceTrackedJLRaceMFSideInfoExperiment,
    "jl_race_mf_sideinfo": SpaceTrackedJLRaceMFSideInfoExperiment,
}


def get_experiment_class(model_name: str) -> Type[BaseExperiment]:
    key = model_name.strip().lower()
    if key not in _EXPERIMENTS:
        available = ", ".join(sorted(_EXPERIMENTS))
        raise KeyError(
            f"No experiment class registered for model '{model_name}'. "
            f"Available models: {available}. Add the model file and register it here."
        )
    return _EXPERIMENTS[key]


def available_models() -> list[str]:
    return sorted(_EXPERIMENTS)
