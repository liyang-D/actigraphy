# readers/geneactive/reader.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..base import (
    BaseDeviceReader,
    default_metadata_path,
    default_raw_output_csv_path,
    get_reader_output_columns,
    save_metadata,
)
from .decode import decode_page
from .header import parse_geneactive_main_header
from .pages import iter_geneactive_pages


def get_output_columns(mode: str) -> list[str]:
    return get_reader_output_columns(mode)


def default_output_csv_path(
    input_path: Path,
    output_dir: Path | None,
    mode: str,
) -> Path:
    get_output_columns(mode)
    return default_raw_output_csv_path(input_path=input_path, output_dir=output_dir)


def update_metadata_for_reader(
    metadata: dict[str, Any],
    output_csv_path: Path,
    mode: str,
    max_pages: int | None,
) -> dict[str, Any]:
    metadata = dict(metadata)

    metadata["reader_output"] = {
        "reader": "geneactive",
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


class GeneActiveReader(BaseDeviceReader):
    name = "geneactive"
    supported_extensions = (".bin",)

    def read(
        self,
        input_path: Path,
        output_csv_path: Path | None = None,
        output_metadata_path: Path | None = None,
        output_dir: Path | None = None,
        mode: str = "full",
        max_pages: int | None = None,
        verbose: bool = False,
    ) -> tuple[Path, Path]:
        return read_geneactive_bin(
            input_path=input_path,
            output_csv_path=output_csv_path,
            output_metadata_path=output_metadata_path,
            output_dir=output_dir,
            mode=mode,
            max_pages=max_pages,
            verbose=verbose,
        )
