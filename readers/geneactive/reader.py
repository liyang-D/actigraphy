from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from models import RawSampleData, coerce_full_sensor_columns
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
from utils import ensure_csv_path, parse_timestamp


DATAFRAME_CHUNK_PAGES = 1000


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
    output_csv_path: Path | None,
    mode: str,
    max_pages: int | None,
) -> dict[str, Any]:
    metadata = dict(metadata)

    metadata["reader_output"] = {
        "reader": "geneactive",
        "mode": mode,
        "columns": get_output_columns(mode),
        "max_pages": max_pages,
    }

    if output_csv_path is not None:
        metadata["reader_output"]["output_file"] = str(output_csv_path)

    return metadata


def page_total_to_process(
    number_of_pages: Any,
    max_pages: int | None,
) -> int | None:
    total_pages = number_of_pages if isinstance(number_of_pages, int) else None

    if total_pages is not None and max_pages is not None:
        return min(total_pages, max_pages)

    if total_pages is not None:
        return total_pages

    return max_pages


def format_page_progress(page_index: int, total_pages: int | None) -> str:
    if total_pages is None:
        return f"{page_index}/?"

    return f"{page_index}/{total_pages}"


def build_samples_dataframe(
    rows: list[dict[str, Any]],
    columns: list[str],
    mode: str,
) -> pd.DataFrame:
    data = pd.DataFrame.from_records(rows, columns=columns)

    if data.empty:
        return data

    data["Time"] = pd.to_datetime(data["Time"].map(parse_timestamp))

    if mode == "full":
        data = coerce_full_sensor_columns(data)

    return data


def read_geneactive_bin(
    input_path: Path,
    output_csv_path: Path | None = None,
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
    else:
        output_csv_path = ensure_csv_path(output_csv_path, "Output CSV path")

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

    total_pages = page_total_to_process(
        number_of_pages=header.decoder_context.get("number_of_pages"),
        max_pages=max_pages,
    )
    processed_pages = 0
    processed_rows = 0

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for page_number, page in enumerate(
            iter_geneactive_pages(
                path=input_path,
                start_line_index=header.next_line_index,
                max_pages=max_pages,
            ),
            start=1,
        ):
            if verbose:
                print(
                    "Decoding page "
                    f"{format_page_progress(page_number, total_pages)} "
                    f"(sequence {page.header.sequence_number}, "
                    f"{page.header.page_time})"
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


def load_geneactive_samples(
    input_path: Path,
    mode: str = "full",
    max_pages: int | None = None,
    verbose: bool = False,
) -> RawSampleData:
    if mode not in {"motion", "full"}:
        raise ValueError("mode must be either 'motion' or 'full'.")

    input_path = Path(input_path)

    if verbose:
        print(f"Loading GENEActiv samples: {input_path}")

    header = parse_geneactive_main_header(input_path)
    metadata = update_metadata_for_reader(
        metadata=header.metadata,
        output_csv_path=None,
        mode=mode,
        max_pages=max_pages,
    )
    columns = get_output_columns(mode)

    total_pages = page_total_to_process(
        number_of_pages=header.decoder_context.get("number_of_pages"),
        max_pages=max_pages,
    )
    chunks: list[pd.DataFrame] = []
    chunk_rows: list[dict[str, Any]] = []
    chunk_page_count = 0
    chunk_number = 0
    decoded_rows = 0
    for page_number, page in enumerate(
        iter_geneactive_pages(
            path=input_path,
            start_line_index=header.next_line_index,
            max_pages=max_pages,
        ),
        start=1,
    ):
        if verbose:
            print(
                "Decoding page "
                f"{format_page_progress(page_number, total_pages)} "
                f"(sequence {page.header.sequence_number}, "
                f"{page.header.page_time})"
            )

        page_rows = list(
            decode_page(
                page=page,
                decoder_context=header.decoder_context,
                mode=mode,
            )
        )
        chunk_rows.extend(page_rows)
        chunk_page_count += 1
        decoded_rows += len(page_rows)

        if chunk_page_count >= DATAFRAME_CHUNK_PAGES:
            chunk_number += 1
            if verbose:
                print(
                    "Building DataFrame chunk "
                    f"{chunk_number} from {len(chunk_rows)} decoded samples"
                )

            chunk_data = build_samples_dataframe(
                rows=chunk_rows,
                columns=columns,
                mode=mode,
            )
            chunks.append(chunk_data)

            if verbose:
                print(f"DataFrame chunk {chunk_number} ready: {len(chunk_data)} rows")

            chunk_rows = []
            chunk_page_count = 0

    if chunk_rows:
        chunk_number += 1
        if verbose:
            print(
                "Building DataFrame chunk "
                f"{chunk_number} from {len(chunk_rows)} decoded samples"
            )

        chunk_data = build_samples_dataframe(
            rows=chunk_rows,
            columns=columns,
            mode=mode,
        )
        chunks.append(chunk_data)

        if verbose:
            print(f"DataFrame chunk {chunk_number} ready: {len(chunk_data)} rows")

    if verbose:
        print(f"Decoded samples: {decoded_rows}")
        print(f"Combining {len(chunks)} DataFrame chunk(s)")

    if chunks:
        data = pd.concat(chunks, ignore_index=True)
    else:
        data = pd.DataFrame(columns=columns)

    if verbose:
        print(f"Sample DataFrame ready: {len(data)} rows")

    sample_rate_hz = header.decoder_context.get("measurement_frequency_hz")
    if sample_rate_hz is None:
        raise ValueError("Missing measurement frequency in GENEActiv header.")

    raw_data = RawSampleData(
        data=data,
        metadata=metadata,
        sample_rate_hz=float(sample_rate_hz),
    )
    raw_data.validate(require_full=(mode == "full"))

    return raw_data


class GeneActiveReader(BaseDeviceReader):
    name = "geneactive"
    supported_extensions = (".bin",)

    def read(
        self,
        input_path: Path,
        output_csv_path: Path | None = None,
        output_dir: Path | None = None,
        mode: str = "full",
        max_pages: int | None = None,
        verbose: bool = False,
    ) -> tuple[Path, Path]:
        return read_geneactive_bin(
            input_path=input_path,
            output_csv_path=output_csv_path,
            output_dir=output_dir,
            mode=mode,
            max_pages=max_pages,
            verbose=verbose,
        )

    def load_samples(
        self,
        input_path: Path,
        mode: str = "full",
        max_pages: int | None = None,
        verbose: bool = False,
    ) -> RawSampleData:
        return load_geneactive_samples(
            input_path=input_path,
            mode=mode,
            max_pages=max_pages,
            verbose=verbose,
        )
