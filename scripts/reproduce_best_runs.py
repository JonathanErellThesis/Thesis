#!/usr/bin/env python
"""Run all configured best runs for registered models."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["youtube", "autorec", "leaf", "unisketchmf"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for model in args.models:
        cfg_path = REPO_ROOT / "config" / "best_runs" / f"{model}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        datasets = args.datasets or sorted((cfg.get("best_runs", {}) or {}).keys())
        for dataset in datasets:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_single.py"),
                "--model", model,
                "--dataset", dataset,
                "--config", str(cfg_path),
                "--data-dir", args.data_dir,
                "--output-dir", args.output_dir,
            ]
            if args.device:
                cmd += ["--device", args.device]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
