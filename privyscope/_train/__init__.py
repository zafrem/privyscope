"""Fine-tuning argument handling and training runners (optional [train] extra)."""

__all__ = ["run_train"]


def __getattr__(name):  # lazy import so the package loads without torch
    if name == "run_train":
        from .runner import run_train

        return run_train
    raise AttributeError(name)
