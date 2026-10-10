"""Load the byte-preserved numeric trainer without global import aliases."""
from __future__ import annotations

import builtins
from functools import lru_cache
import importlib
import importlib.util
from pathlib import Path
import sys
import types


@lru_cache(maxsize=1)
def load_trainer() -> types.ModuleType:
    """Return the frozen trainer with its flat imports scoped to its own module.

    The archived script imports ``datasets``, ``models``, ``loss`` and
    ``phase4_utils``. Redirect only its import calls to ``my_experiments``;
    existing third-party modules under those names and ``sys.path`` remain
    untouched. A local view of ``sys.path`` absorbs its original path insertion.
    """
    package = importlib.import_module("my_experiments.models")
    root = Path(package.__file__).resolve().parents[1]
    path = root / "scripts" / "train" / "09_train_experiment.py"
    name = "sagf_dti._frozen_trainer"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the packaged frozen trainer at {path}")
    trainer = importlib.util.module_from_spec(spec)
    local_sys = types.ModuleType("sys")
    local_sys.__dict__.update(sys.__dict__)
    local_sys.path = list(sys.path)
    legacy_names = {"datasets", "models", "loss", "phase4_utils"}

    def scoped_import(module_name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and module_name == "sys":
            return local_sys
        if level == 0 and module_name.split(".", 1)[0] in legacy_names:
            qualified = "my_experiments." + module_name
            module = importlib.import_module(qualified)
            if fromlist:
                return module
            return importlib.import_module("my_experiments." + module_name.split(".", 1)[0])
        return builtins.__import__(module_name, globals, locals, fromlist, level)

    trainer.__dict__["__builtins__"] = {**vars(builtins), "__import__": scoped_import}
    sys.modules[name] = trainer
    try:
        spec.loader.exec_module(trainer)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return trainer
