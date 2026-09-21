"""Read tensor-only release checkpoints and older wrapped checkpoints."""

from collections.abc import Mapping

import torch


def model_state(checkpoint, name):
    """Select one model from a flat state dict or a legacy named weight group."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a checkpoint state dictionary.")
    if name in checkpoint:
        state = checkpoint[name]
    else:
        prefix = name + "."
        grouped = {
            key[len(prefix) :]: value
            for key, value in checkpoint.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        state = grouped or checkpoint
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        )
    ):
        raise ValueError("Missing tensor weights for " + name)
    return state
