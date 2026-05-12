from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


IndexSelector = slice | list[int]
RunningMetric = dict[str, float | int | None]


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


def clean_cell(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def parse_numeric(value: str) -> float | None:
    try:
        parsed = float(clean_cell(value))
    except ValueError:
        return None

    if math.isnan(parsed):
        return None

    return parsed


def validate_streamable_row_selector(row_selector: IndexSelector) -> None:
    if isinstance(row_selector, list):
        if any(index < 0 for index in row_selector):
            raise ValueError("Negative row indexes are not supported.")
        return

    if row_selector.step not in (None, 1):
        raise ValueError("Row ranges with steps are not supported.")

    if row_selector.start is not None and row_selector.start < 0:
        raise ValueError("Negative row range starts are not supported.")

    if row_selector.stop is not None and row_selector.stop < 0:
        raise ValueError("Negative row range ends are not supported.")


def row_is_selected(row_index: int, row_selector: IndexSelector) -> bool:
    if isinstance(row_selector, list):
        return row_index in row_selector

    start = row_selector.start or 0
    end = row_selector.stop

    if row_index < start:
        return False

    return end is None or row_index < end


def row_selection_complete(row_index: int, row_selector: IndexSelector) -> bool:
    if isinstance(row_selector, list):
        return bool(row_selector) and row_index > max(row_selector)

    return row_selector.stop is not None and row_index >= row_selector.stop


def select_columns(row: list[str], col_selector: IndexSelector) -> list[str]:
    if isinstance(col_selector, list):
        return [row[index] for index in col_selector]

    return row[col_selector]


def load_csv_range(
    path: Path,
    row_range: str | None = None,
    col_range: str | None = None,
) -> pd.DataFrame:
    row_selector = parse_range(row_range)
    col_selector = parse_range(col_range)
    validate_streamable_row_selector(row_selector)

    rows: list[list[str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row_index, row in enumerate(reader):
            if row_selection_complete(row_index, row_selector):
                break

            if not row_is_selected(row_index, row_selector):
                continue

            rows.append(select_columns(row, col_selector))

    return pd.DataFrame(rows)


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


def init_running_metric() -> RunningMetric:
    return {
        "count": 0,
        "sum_error": 0.0,
        "sum_absolute_error": 0.0,
        "sum_squared_error": 0.0,
        "max_absolute_error": None,
    }


def update_running_metric(metric: RunningMetric, error: float) -> None:
    absolute_error = abs(error)

    metric["count"] = int(metric["count"]) + 1
    metric["sum_error"] = float(metric["sum_error"]) + error
    metric["sum_absolute_error"] = float(metric["sum_absolute_error"]) + absolute_error
    metric["sum_squared_error"] = float(metric["sum_squared_error"]) + error**2

    max_absolute_error = metric["max_absolute_error"]
    if max_absolute_error is None or absolute_error > float(max_absolute_error):
        metric["max_absolute_error"] = absolute_error


def finalize_running_metric(metric: RunningMetric) -> dict[str, Any]:
    count = int(metric["count"])

    if count == 0:
        return {
            "count": 0,
            "mean_error": None,
            "mean_absolute_error": None,
            "root_mean_squared_error": None,
            "max_absolute_error": None,
            "std_error": None,
        }

    mean_error = float(metric["sum_error"]) / count
    mean_squared_error = float(metric["sum_squared_error"]) / count
    variance = max(mean_squared_error - mean_error**2, 0.0)

    return {
        "count": count,
        "mean_error": mean_error,
        "mean_absolute_error": float(metric["sum_absolute_error"]) / count,
        "root_mean_squared_error": math.sqrt(mean_squared_error),
        "max_absolute_error": metric["max_absolute_error"],
        "std_error": math.sqrt(variance),
    }


def compare_row_values(
    reference_row: list[str],
    candidate_row: list[str],
    per_column_metrics: list[RunningMetric],
    overall_metric: RunningMetric,
) -> None:
    column_count = len(per_column_metrics)

    if len(reference_row) < column_count:
        raise ValueError(
            "Reference row has fewer columns than the candidate row. "
            f"Reference columns: {len(reference_row)}, candidate columns: {column_count}."
        )

    if len(candidate_row) < column_count:
        raise ValueError(
            "Candidate row has fewer columns than the first candidate data row. "
            f"Row columns: {len(candidate_row)}, expected columns: {column_count}."
        )

    for column_index in range(column_count):
        reference_value = parse_numeric(reference_row[column_index])
        candidate_value = parse_numeric(candidate_row[column_index])

        if reference_value is None or candidate_value is None:
            continue

        error = candidate_value - reference_value
        update_running_metric(per_column_metrics[column_index], error)
        update_running_metric(overall_metric, error)


def finalize_stream_summary(
    rows: int,
    columns: int,
    per_column_metrics: list[RunningMetric],
    overall_metric: RunningMetric,
    reference_start_row: int,
    candidate_start_row: int,
    start_time: str,
) -> dict[str, Any]:
    overall = finalize_running_metric(overall_metric)

    if overall["count"] == 0:
        raise ValueError("No comparable numeric values found in the aligned rows.")

    per_column = []
    for column_index, metric in enumerate(per_column_metrics):
        column_summary = finalize_running_metric(metric)
        column_summary["column_index"] = column_index
        per_column.append(column_summary)

    return {
        "alignment": {
            "mode": "candidate_timestamp",
            "time_column_index": 0,
            "start_time": start_time,
            "reference_start_row": reference_start_row,
            "candidate_start_row": candidate_start_row,
        },
        "shape": {
            "rows": rows,
            "columns": columns,
        },
        "overall": overall,
        "per_column": per_column,
    }


def compare_by_candidate_timestamp(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as candidate_file:
        candidate_reader = csv.reader(candidate_file)
        next(candidate_reader, None)

        candidate_first_row = next(
            (row for row in candidate_reader if row),
            None,
        )

        if candidate_first_row is None:
            raise ValueError("Candidate CSV does not contain any data rows.")

        start_time = clean_cell(candidate_first_row[0])
        column_count = len(candidate_first_row)
        per_column_metrics = [init_running_metric() for _ in range(column_count)]
        overall_metric = init_running_metric()

        with reference_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as reference_file:
            reference_reader = csv.reader(reference_file)
            reference_start_row = None
            reference_first_row = None

            for row_index, row in enumerate(reference_reader):
                if row and clean_cell(row[0]) == start_time:
                    reference_start_row = row_index
                    reference_first_row = row
                    break

            if reference_start_row is None or reference_first_row is None:
                raise ValueError(
                    "Could not find the candidate start timestamp in the reference CSV: "
                    f"{start_time}"
                )

            rows_compared = 0
            compare_row_values(
                reference_row=reference_first_row,
                candidate_row=candidate_first_row,
                per_column_metrics=per_column_metrics,
                overall_metric=overall_metric,
            )
            rows_compared += 1

            for candidate_row in candidate_reader:
                if not candidate_row:
                    continue

                try:
                    reference_row = next(reference_reader)
                except StopIteration as exc:
                    raise ValueError(
                        "Reference CSV ended before all candidate rows were compared."
                    ) from exc

                compare_row_values(
                    reference_row=reference_row,
                    candidate_row=candidate_row,
                    per_column_metrics=per_column_metrics,
                    overall_metric=overall_metric,
                )
                rows_compared += 1

    return finalize_stream_summary(
        rows=rows_compared,
        columns=column_count,
        per_column_metrics=per_column_metrics,
        overall_metric=overall_metric,
        reference_start_row=reference_start_row,
        candidate_start_row=1,
        start_time=start_time,
    )


def save_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_summary(summary: dict[str, Any]) -> None:
    print("Comparison summary")
    print("==================")

    alignment = summary.get("alignment")
    if alignment is not None:
        print(f"Alignment mode: {alignment['mode']}")
        print(f"Start time: {alignment['start_time']}")
        print(f"Reference start row: {alignment['reference_start_row']}")
        print(f"Candidate start row: {alignment['candidate_start_row']}")

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
            "position. If no ranges are supplied, the candidate header row is "
            "skipped and the reference is aligned by the first candidate "
            "timestamp."
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
        default=None,
        help="Optional reference row range, e.g. '0:1000', '1:', or ':'.",
    )
    parser.add_argument(
        "--candidate-rows",
        type=str,
        default=None,
        help="Optional candidate row range, e.g. '0:1000', '1:', or ':'.",
    )
    parser.add_argument(
        "--reference-cols",
        type=str,
        default=None,
        help="Optional reference column range, e.g. '1:4'.",
    )
    parser.add_argument(
        "--candidate-cols",
        type=str,
        default=None,
        help="Optional candidate column range, e.g. '1:4'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for the comparison summary.",
    )

    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    has_manual_range = any(
        value is not None
        for value in (
            args.reference_rows,
            args.candidate_rows,
            args.reference_cols,
            args.candidate_cols,
        )
    )

    if not has_manual_range:
        return compare_by_candidate_timestamp(
            reference_path=args.reference,
            candidate_path=args.candidate,
        )

    reference_selected = load_csv_range(
        path=args.reference,
        row_range=args.reference_rows,
        col_range=args.reference_cols,
    )
    candidate_selected = load_csv_range(
        path=args.candidate,
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
