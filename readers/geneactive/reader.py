# readers/geneactive/reader.py

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .decode import decode_page
from .header import parse_geneactive_main_header, save_metadata
from .pages import iter_geneactive_pages


DEFAULT_TEST_BIN = Path(
    "data_test/efthyvoulos__111238_2026-03-06 14-16-51.bin"
)


def get_output_columns(mode: str) -> list[str]:
    if mode == "motion":
        return ["Time", "Ax", "Ay", "Az"]

    if mode == "full":
        return ["Time", "Ax", "Ay", "Az", "Lux", "Button", "Temperature"]

    raise ValueError("mode must be either 'motion' or 'full'.")


def default_output_csv_path(input_path: Path, output_dir: Path | None, mode: str) -> Path:
    output_base = input_path.stem

    if output_dir is None:
        output_dir = input_path.parent

    suffix = "_raw" if mode == "full" else "_motion"

    return output_dir / f"{output_base}{suffix}.csv"


def default_metadata_path(output_csv_path: Path) -> Path:
    return output_csv_path.with_suffix(".metadata.json")


def update_metadata_for_reader(
    metadata: dict[str, Any],
    output_csv_path: Path,
    mode: str,
    max_pages: int | None,
) -> dict[str, Any]:
    metadata = dict(metadata)

    metadata["reader_output"] = {
        "output_file": str(output_csv_path),
        "mode": mode,
        "columns": get_output_columns(mode),
        "max_pages": max_pages,
    }

    return metadata


def read_geneactive_bin(
    input_path: Path,
    output_csv_path: Path | None = None,
    output_metadata_path: Path | None = None,
    output_dir: Path | None = None,
    mode: str = "full",
    max_pages: int | None = None,
    verbose: bool = False,
) -> tuple[Path, Path]:
    if mode not in {"motion", "full"}:
        raise ValueError("mode must be either 'motion' or 'full'.")

    input_path = Path(input_path)

    if output_csv_path is None:
        output_csv_path = default_output_csv_path(
            input_path=input_path,
            output_dir=output_dir,
            mode=mode,
        )

    if output_metadata_path is None:
        output_metadata_path = default_metadata_path(output_csv_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Reading GENEActiv file: {input_path}")

    header = parse_geneactive_main_header(input_path)

    metadata = update_metadata_for_reader(
        metadata=header.metadata,
        output_csv_path=output_csv_path,
        mode=mode,
        max_pages=max_pages,
    )

    save_metadata(metadata, output_metadata_path)

    columns = get_output_columns(mode)

    processed_pages = 0
    processed_rows = 0

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for page in iter_geneactive_pages(
            path=input_path,
            start_line_index=header.next_line_index,
            max_pages=max_pages,
        ):
            if verbose:
                print(
                    "Decoding page "
                    f"{page.header.sequence_number} "
                    f"at {page.header.page_time}"
                )

            for row in decode_page(
                page=page,
                decoder_context=header.decoder_context,
                mode=mode,
            ):
                writer.writerow(row)
                processed_rows += 1

            processed_pages += 1

    if verbose:
        print(f"CSV saved to: {output_csv_path}")
        print(f"Metadata saved to: {output_metadata_path}")
        print(f"Pages processed: {processed_pages}")
        print(f"Rows written: {processed_rows}")

    return output_csv_path, output_metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a GENEActiv .bin file and export sample-level CSV data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TEST_BIN,
        help="Input GENEActiv .bin file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, a default name is generated.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Output metadata JSON path. If omitted, a default name is generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Used only when --output is omitted.",
    )
    parser.add_argument(
        "--mode",
        choices=["motion", "full"],
        default="full",
        help="Reader output mode.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help=(
            "Maximum number of pages to process. "
            "Use 1 by default for module-level testing. "
            "Pass 0 to process all pages."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_pages = None if args.max_pages == 0 else args.max_pages

    read_geneactive_bin(
        input_path=args.input,
        output_csv_path=args.output,
        output_metadata_path=args.metadata,
        output_dir=args.output_dir,
        mode=args.mode,
        max_pages=max_pages,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()