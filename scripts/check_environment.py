#!/usr/bin/env python3
"""Check the cached-feature runtime without importing RDKit or external models."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

EXPECTED = {
    "torch": "2.7.1", "torch-geometric": "2.6.1", "torch-scatter": "2.1.2",
    "torch-sparse": "0.6.18", "numpy": "1.26.4", "pandas": "2.2.3",
    "scikit-learn": "1.5.2", "PyYAML": "6.0.2",
}
IMPORTS = {
    "torch": "torch", "torch-geometric": "torch_geometric", "torch-scatter": "torch_scatter",
    "torch-sparse": "torch_sparse", "numpy": "numpy", "pandas": "pandas",
    "scikit-learn": "sklearn", "PyYAML": "yaml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strict-versions", action="store_true", help="Fail when a release pin differs.")
    parser.add_argument("--output", type=Path, help="Write the JSON evidence to this path.")
    args = parser.parse_args()
    report = {"python": sys.version, "platform": platform.platform(), "packages": {},
              "checks": {}, "warnings": [], "errors": []}
    modules = {}
    for distribution, module_name in IMPORTS.items():
        try:
            version = importlib.metadata.version(distribution)
            report["packages"][distribution] = version
            modules[module_name] = importlib.import_module(module_name)
            if version.split("+")[0] != EXPECTED[distribution]:
                message = f"{distribution}: installed {version}; release pin {EXPECTED[distribution]}"
                report["errors" if args.strict_versions else "warnings"].append(message)
        except Exception as exc:
            report["errors"].append(f"{distribution}: {type(exc).__name__}: {exc}")
    try:
        report["packages"]["rdkit"] = importlib.metadata.version("rdkit")
    except importlib.metadata.PackageNotFoundError:
        report["packages"]["rdkit"] = None
    report["checks"]["rdkit_imported"] = False
    if "torch" in modules:
        torch = modules["torch"]
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_architectures"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
        report["device"] = device
        if device == "cuda" and not torch.cuda.is_available():
            report["errors"].append("CUDA was requested but torch.cuda.is_available() is false")
        else:
            try:
                if device == "cuda":
                    props = torch.cuda.get_device_properties(0)
                    report["gpu"] = {"name": props.name, "capability": [props.major, props.minor],
                                     "total_memory_bytes": props.total_memory}
                x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
                (x @ x).sum().backward()
                assert torch.isfinite(x.grad).all().item()
                report["checks"]["torch_forward_backward"] = True
                scatter = modules.get("torch_scatter")
                sparse = modules.get("torch_sparse")
                if scatter:
                    out = scatter.scatter_add(torch.tensor([1., 2., 3.], device=device),
                                              torch.tensor([0, 1, 0], device=device), dim=0)
                    assert torch.equal(out, torch.tensor([4., 2.], device=device))
                    report["checks"]["scatter"] = True
                if sparse:
                    index = torch.tensor([[0, 1], [0, 1]], device=device)
                    values = torch.ones(2, device=device)
                    out = sparse.spmm(index, values, 2, 2, x.detach())
                    assert torch.equal(out, x.detach())
                    report["checks"]["sparse"] = True
                if device == "cuda":
                    torch.cuda.synchronize()
            except Exception as exc:
                report["errors"].append(f"runtime smoke check: {type(exc).__name__}: {exc}")
    report["status"] = "passed" if not report["errors"] else "failed"
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
