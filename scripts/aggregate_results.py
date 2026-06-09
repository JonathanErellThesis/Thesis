#!/usr/bin/env python
"""Aggregate saved run metrics into a CSV table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--out", default="outputs/summary.csv")
    args = parser.parse_args()

    rows = []
    for metrics_path in Path(args.output_dir).glob("*/metrics.json"):
        run_name = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"run": run_name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("run") if rows else pd.DataFrame()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
