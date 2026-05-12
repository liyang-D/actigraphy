from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


IndexSelector = slice | list[int]


def parse_range(range_str: str | None) -> IndexSelector:
    """
    Parse a Python-style row or column selector.

    Examples:
        None      -> all rows/columns
        ":"       -> all rows/columns
        "0:100"   -> rows/columns 0 through 99
        "100:"    -> rows/columns 100 onward
        ":100"    -> rows/columns up to 99
        "5"       -> row/column 5 only
    """
    if range_str is None:
        return slice(None)

    range_str = range_str.strip()
    if not range_str or range_str == ":":
        return slice(None)

    if ":" not in range_str:
        return [int(range_str)]

    start_str, end_str = range_str.split(":", 1)
    start = int(start_str) if start_str.strip() else None
    end = int(end_str) if end_str.strip() else None

    return slice(start, end)


def load_csv_raw(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    return pd.DataFrame(rows)


def select_range(
    df: pd.DataFrame,
    row_range: str | None = None,
    col_range: str | None = None,
) -> pd.DataFrame:
    row_selector = parse_range(row_range)
    col_selector = parse_range(col_range)

    return df.iloc[row_selector, col_selector].copy()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _numeric_values(df: pd.DataFrame) -> np.ndarray:
    return coerce_numeric(df).to_numpy(dtype=float)


def compare_dataframes(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            "Selected ranges must have the same shape. "
            f"Reference shape: {reference.shape}, candidate shape: {candidate.shape}"
        )

    ref_values = _numeric_values(reference)
    cand_values = _numeric_values(candidate)

    diff = cand_values - ref_values
    abs_diff = np.abs(diff)
    valid_mask = ~np.isnan(ref_values) & ~np.isnan(cand_values)

    if not np.any(valid_mask):
        raise ValueError("No comparable numeric values found in the selected ranges.")

    valid_diff = diff[valid_mask]
    valid_abs_diff = abs_diff[valid_mask]

    per_column: list[dict[str, Any]] = []
    for column_index in range(reference.shape[1]):
        ref_col = ref_values[:, column_index]
        cand_col = cand_values[:, column_index]
        col_mask = ~np.isnan(ref_col) & ~np.isnan(cand_col)

        if not np.any(col_mask):
            per_column.append(
                {
                    "column_index": column_index,
                    "count": 0,
                    "mean_error": None,
                    "mean_absolute_error": None,
                    "root_mean_squared_error": None,
                    "max_absolute_error": None,
                    "std_error": None,
                }
            )
            continue

        col_diff = cand_col[col_mask] - ref_col[col_mask]
        col_abs_diff = np.abs(col_diff)

        per_column.append(
            {
                "column_index": column_index,
                "count": int(col_mask.sum()),
                "mean_error": float(np.mean(col_diff)),
                "mean_absolute_error": float(np.mean(col_abs_diff)),
                "root_mean_squared_error": float(np.sqrt(np.mean(col_diff**2))),
                "max_absolute_error": float(np.max(col_abs_diff)),
                "std_error": float(np.std(col_diff)),
            }
        )

    return {
        "shape": {
            "rows": int(reference.shape[0]),
            "columns": int(reference.shape[1]),
        },
        "overall": {
            "count": int(valid_mask.sum()),
            "mean_error": float(np.mean(valid_diff)),
            "mean_absolute_error": float(np.mean(valid_abs_diff)),
            "root_mean_squared_error": float(np.sqrt(np.mean(valid_diff**2))),
            "max_absolute_error": float(np.max(valid_abs_diff)),
            "std_error": float(np.std(valid_diff)),
        },
        "per_column": per_column,
    }


def save_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_summary(summary: dict[str, Any]) -> None:
    print("Comparison summary")
    print("==================")
    print(f"Rows: {summary['shape']['rows']}")
    print(f"Columns: {summary['shape']['columns']}")

    overall = summary["overall"]
    print("\nOverall")
    print(f"Count: {overall['count']}")
    print(f"Mean error: {overall['mean_error']}")
    print(f"Mean absolute error: {overall['mean_absolute_error']}")
    print(f"RMSE: {overall['root_mean_squared_error']}")
    print(f"Max absolute error: {overall['max_absolute_error']}")
    print(f"Std error: {overall['std_error']}")

    print("\nPer column")
    for column in summary["per_column"]:
        print(
            f"Column {column['column_index']}: "
            f"count={column['count']}, "
            f"MAE={column['mean_absolute_error']}, "
            f"RMSE={column['root_mean_squared_error']}, "
            f"max_abs={column['max_absolute_error']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected numeric ranges from two CSV files by row/column "
            "position. Headers are not interpreted semantically."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference CSV path.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate CSV path.",
    )
    parser.add_argument(
        "--reference-rows",
        type=str,
        required=True,
        help="Reference row range, e.g. '0:1000', '1:', or ':'.",
    )
    parser.add_argument(
        "--candidate-rows",
        type=str,
        required=True,
        help="Candidate row range, e.g. '0:1000', '1:', or ':'.",
    )
    parser.add_argument(
        "--reference-cols",
        type=str,
        required=True,
        help="Reference column range, e.g. '1:4'.",
    )
    parser.add_argument(
        "--candidate-cols",
        type=str,
        required=True,
        help="Candidate column range, e.g. '1:4'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for the comparison summary.",
    )

    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_csv_raw(path=args.reference)
    candidate = load_csv_raw(path=args.candidate)

    reference_selected = select_range(
        reference,
        row_range=args.reference_rows,
        col_range=args.reference_cols,
    )
    candidate_selected = select_range(
        candidate,
        row_range=args.candidate_rows,
        col_range=args.candidate_cols,
    )

    return compare_dataframes(reference_selected, candidate_selected)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run(args)
    except (OSError, ValueError, IndexError) as exc:
        parser.error(str(exc))

    print_summary(summary)

    if args.output is not None:
        save_summary(summary, args.output)
        print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
