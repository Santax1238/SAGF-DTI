"""Standalone public API for the original SAGF-DTI Gaussian model."""

__version__ = "0.1.0"
__all__ = ["SAGFDTI"]


def __getattr__(name):
    # Config validation and help remain usable before installing the GPU runtime.
    if name == "SAGFDTI":
        from .model import SAGFDTI

        return SAGFDTI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
