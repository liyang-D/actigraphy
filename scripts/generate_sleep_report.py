from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from reports import generate_sleep_report_pdf
from reports.sleep_report import (
    DEFAULT_DAY_START_HOUR,
    DEFAULT_DAYS_PER_PAGE,
    DEFAULT_LUX_LOG_SCALE_MAX,
    DEFAULT_TITLE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a simple Actigraphy Sleep Report PDF from a standard "
            "pipeline CSV. SVM-only files produce SVM plots; summary files "
            "also include available light on a log10 scale."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV with standard headers, preferably a Step 1B output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PDF path. Defaults to <input>_sleep_report.pdf.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Report title.",
    )
    parser.add_argument(
        "--report-date",
        default=None,
        help="Optional report date text. Defaults to today's date.",
    )
    parser.add_argument(
        "--day-start-hour",
        type=int,
        default=DEFAULT_DAY_START_HOUR,
        help="Hour used as the start of each 24-hour report window.",
    )
    parser.add_argument(
        "--days-per-page",
        type=int,
        default=DEFAULT_DAYS_PER_PAGE,
        help="Number of daily charts per PDF page after the title page.",
    )
    parser.add_argument(
        "--activity-scale",
        type=float,
        default=None,
        help=(
            "Optional activity y-axis maximum. Defaults to the 99th percentile "
            "of SVM_sum across the full file."
        ),
    )
    parser.add_argument(
        "--lux-log-scale-max",
        type=float,
        default=DEFAULT_LUX_LOG_SCALE_MAX,
        help="Maximum for the log10(lux + 1) y-axis. Defaults to 5.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print report generation progress.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = generate_sleep_report_pdf(
            input_csv_path=args.input,
            output_pdf_path=args.output,
            title=args.title,
            report_date=args.report_date,
            day_start_hour=args.day_start_hour,
            days_per_page=args.days_per_page,
            activity_scale=args.activity_scale,
            lux_log_scale_max=args.lux_log_scale_max,
            verbose=args.verbose,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if not args.verbose:
        print(f"Sleep report saved to: {output_path}")


if __name__ == "__main__":
    main()
