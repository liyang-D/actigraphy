from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ERROR_METRICS = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "max_absolute_error",
)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    required_keys = {"shape", "overall", "per_column"}
    missing_keys = required_keys - set(summary)
    if missing_keys:
        raise ValueError(f"{path} is missing keys: {sorted(missing_keys)}")

    return summary


def format_value(value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.10g}"

    return str(value)


def smaller_error_label(left_value: Any, right_value: Any, left_label: str, right_label: str) -> str:
    if left_value is None or right_value is None:
        return "-"

    if left_value == right_value:
        return "tie"

    return left_label if left_value < right_value else right_label


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    text_rows = [[format_value(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]

    def line(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print(line(headers))
    print("-+-".join("-" * width for width in widths))

    for row in text_rows:
        print(line(row))


def column_metrics_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {column["column_name"]: column for column in summary["per_column"]}


def ordered_column_names(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> list[str]:
    names = list(left_summary.get("columns", []))

    for column in right_summary.get("columns", []):
        if column not in names:
            names.append(column)

    if names:
        return names

    left_columns = column_metrics_by_name(left_summary)
    right_columns = column_metrics_by_name(right_summary)
    return list(dict.fromkeys([*left_columns.keys(), *right_columns.keys()]))


def build_overall_rows(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    left_label: str,
    right_label: str,
) -> list[list[Any]]:
    rows = [
        ["rows", left_summary["shape"].get("rows"), right_summary["shape"].get("rows"), "-"],
        [
            "columns",
            left_summary["shape"].get("columns"),
            right_summary["shape"].get("columns"),
            "-",
        ],
        [
            "count",
            left_summary["overall"].get("count"),
            right_summary["overall"].get("count"),
            "-",
        ],
    ]

    for metric in ERROR_METRICS:
        left_value = left_summary["overall"].get(metric)
        right_value = right_summary["overall"].get(metric)
        rows.append(
            [
                metric,
                left_value,
                right_value,
                smaller_error_label(left_value, right_value, left_label, right_label),
            ]
        )

    return rows


def build_column_rows(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    left_label: str,
    right_label: str,
) -> list[list[Any]]:
    left_columns = column_metrics_by_name(left_summary)
    right_columns = column_metrics_by_name(right_summary)

    rows: list[list[Any]] = []
    for column_name in ordered_column_names(left_summary, right_summary):
        left_column = left_columns.get(column_name, {})
        right_column = right_columns.get(column_name, {})
        left_mae = left_column.get("mean_absolute_error")
        right_mae = right_column.get("mean_absolute_error")

        rows.append(
            [
                column_name,
                left_column.get("count"),
                right_column.get("count"),
                left_mae,
                right_mae,
                left_column.get("root_mean_squared_error"),
                right_column.get("root_mean_squared_error"),
                left_column.get("max_absolute_error"),
                right_column.get("max_absolute_error"),
                smaller_error_label(left_mae, right_mae, left_label, right_label),
            ]
        )

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two JSON summaries produced by scripts.compare_csv."
    )
    parser.add_argument(
        "--left",
        type=Path,
        required=True,
        help="First comparison JSON file.",
    )
    parser.add_argument(
        "--right",
        type=Path,
        required=True,
        help="Second comparison JSON file.",
    )
    parser.add_argument(
        "--left-label",
        default="left",
        help="Display label for the first JSON file.",
    )
    parser.add_argument(
        "--right-label",
        default="right",
        help="Display label for the second JSON file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    left_summary = load_summary(args.left)
    right_summary = load_summary(args.right)

    print("Overall")
    print_table(
        ["metric", args.left_label, args.right_label, "smaller_error"],
        build_overall_rows(
            left_summary,
            right_summary,
            args.left_label,
            args.right_label,
        ),
    )

    print("\nPer column")
    print_table(
        [
            "column",
            f"{args.left_label}_count",
            f"{args.right_label}_count",
            f"{args.left_label}_MAE",
            f"{args.right_label}_MAE",
            f"{args.left_label}_RMSE",
            f"{args.right_label}_RMSE",
            f"{args.left_label}_max",
            f"{args.right_label}_max",
            "smaller_MAE",
        ],
        build_column_rows(
            left_summary,
            right_summary,
            args.left_label,
            args.right_label,
        ),
    )


if __name__ == "__main__":
    main()
