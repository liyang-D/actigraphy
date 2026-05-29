from __future__ import annotations

import csv
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from models import RawSampleData, coerce_full_sensor_columns
from ..base import (
    BaseDeviceReader,
    default_metadata_path,
    default_raw_output_csv_path,
    get_reader_output_columns,
    save_metadata,
)
from .decode import decode_page_columns
from .header import parse_geneactive_main_header
from .pages import iter_geneactive_pages
from .models import GeneActivePage
from utils import ensure_csv_path, format_csv_value, parse_timestamp, validate_output_precision


DATAFRAME_CHUNK_PAGES = 1000
DEFAULT_DECODE_WORKERS = 1
ColumnBatch = dict[str, list[Any]]


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
    workers: int,
) -> dict[str, Any]:
    metadata = dict(metadata)

    metadata["reader_output"] = {
        "reader": "geneactive",
        "mode": mode,
        "columns": get_output_columns(mode),
        "max_pages": max_pages,
        "workers": workers,
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


def format_page_chunk_progress(
    start_page: int,
    end_page: int,
    total_pages: int | None,
) -> str:
    if total_pages is None:
        return f"{start_page}-{end_page}/?"

    return f"{start_page}-{end_page}/{total_pages}"


def merge_column_batches(
    batches: list[ColumnBatch],
    columns: list[str],
) -> ColumnBatch:
    merged = {column: [] for column in columns}

    for batch in batches:
        for column in columns:
            merged[column].extend(batch[column])

    return merged


def decode_page_chunk_to_columns(
    pages: list[GeneActivePage],
    decoder_context: dict[str, Any],
    mode: str,
    columns: list[str],
) -> ColumnBatch:
    page_batches = [
        decode_page_columns(
            page=page,
            decoder_context=decoder_context,
            mode=mode,
        )
        for page in pages
    ]

    return merge_column_batches(page_batches, columns)


def iter_page_chunks(
    input_path: Path,
    start_line_index: int,
    max_pages: int | None,
    chunk_pages: int,
) -> Iterator[tuple[int, int, list[GeneActivePage]]]:
    chunk: list[GeneActivePage] = []
    chunk_start_page = 1

    for page_number, page in enumerate(
        iter_geneactive_pages(
            path=input_path,
            start_line_index=start_line_index,
            max_pages=max_pages,
        ),
        start=1,
    ):
        if not chunk:
            chunk_start_page = page_number

        chunk.append(page)

        if len(chunk) >= chunk_pages:
            yield chunk_start_page, page_number, chunk
            chunk = []

    if chunk:
        yield chunk_start_page, chunk_start_page + len(chunk) - 1, chunk


def iter_decoded_column_chunks(
    input_path: Path,
    start_line_index: int,
    decoder_context: dict[str, Any],
    mode: str,
    columns: list[str],
    max_pages: int | None,
    total_pages: int | None,
    workers: int,
    verbose: bool,
) -> Iterator[tuple[int, int, ColumnBatch]]:
    page_chunks = iter_page_chunks(
        input_path=input_path,
        start_line_index=start_line_index,
        max_pages=max_pages,
        chunk_pages=DATAFRAME_CHUNK_PAGES,
    )

    if workers == 1:
        for start_page, end_page, pages in page_chunks:
            if verbose:
                print(
                    "Decoding page chunk "
                    f"{format_page_chunk_progress(start_page, end_page, total_pages)}"
                )

            yield (
                start_page,
                end_page,
                decode_page_chunk_to_columns(
                    pages=pages,
                    decoder_context=decoder_context,
                    mode=mode,
                    columns=columns,
                ),
            )
        return

    pending: deque[tuple[int, int, Future[ColumnBatch]]] = deque()

    def submit_next(executor: ProcessPoolExecutor) -> bool:
        try:
            start_page, end_page, pages = next(page_chunks)
        except StopIteration:
            return False

        if verbose:
            print(
                "Submitting page chunk "
                f"{format_page_chunk_progress(start_page, end_page, total_pages)}"
            )

        future = executor.submit(
            decode_page_chunk_to_columns,
            pages,
            decoder_context,
            mode,
            columns,
        )
        pending.append((start_page, end_page, future))
        return True

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers * 2):
            if not submit_next(executor):
                break

        while pending:
            start_page, end_page, future = pending.popleft()
            batch = future.result()

            if verbose:
                print(
                    "Decoded page chunk "
                    f"{format_page_chunk_progress(start_page, end_page, total_pages)}"
                )

            yield start_page, end_page, batch
            submit_next(executor)


def build_samples_dataframe(
    column_batch: ColumnBatch,
    columns: list[str],
    mode: str,
) -> pd.DataFrame:
    data = pd.DataFrame(column_batch, columns=columns)

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
    workers: int = DEFAULT_DECODE_WORKERS,
    precision: int = 5,
    verbose: bool = False,
) -> tuple[Path, Path]:
    if mode not in {"motion", "full"}:
        raise ValueError("mode must be either 'motion' or 'full'.")

    precision = validate_output_precision(precision)
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
        workers=workers,
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
        writer = csv.writer(f)
        writer.writerow(columns)

        for start_page, end_page, column_batch in iter_decoded_column_chunks(
            input_path=input_path,
            start_line_index=header.next_line_index,
            decoder_context=header.decoder_context,
            mode=mode,
            columns=columns,
            max_pages=max_pages,
            total_pages=total_pages,
            workers=workers,
            verbose=verbose,
        ):
            rows = zip(*(column_batch[column] for column in columns))
            writer.writerows(
                [format_csv_value(value, precision) for value in row]
                for row in rows
            )
            processed_pages += end_page - start_page + 1
            processed_rows += len(column_batch["Time"])

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
    workers: int = DEFAULT_DECODE_WORKERS,
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
        workers=workers,
    )
    columns = get_output_columns(mode)

    total_pages = page_total_to_process(
        number_of_pages=header.decoder_context.get("number_of_pages"),
        max_pages=max_pages,
    )
    chunks: list[pd.DataFrame] = []
    chunk_number = 0
    decoded_rows = 0
    for start_page, end_page, column_batch in iter_decoded_column_chunks(
        input_path=input_path,
        start_line_index=header.next_line_index,
        decoder_context=header.decoder_context,
        mode=mode,
        columns=columns,
        max_pages=max_pages,
        total_pages=total_pages,
        workers=workers,
        verbose=verbose,
    ):
        row_count = len(column_batch["Time"])
        decoded_rows += row_count
        chunk_number += 1

        if verbose:
            print(
                "Building DataFrame chunk "
                f"{chunk_number} from {row_count} decoded samples "
                f"(pages {format_page_chunk_progress(start_page, end_page, total_pages)})"
            )

        chunk_data = build_samples_dataframe(
            column_batch=column_batch,
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
        workers: int = DEFAULT_DECODE_WORKERS,
        precision: int = 5,
        verbose: bool = False,
    ) -> tuple[Path, Path]:
        return read_geneactive_bin(
            input_path=input_path,
            output_csv_path=output_csv_path,
            output_dir=output_dir,
            mode=mode,
            max_pages=max_pages,
            workers=workers,
            precision=precision,
            verbose=verbose,
        )

    def load_samples(
        self,
        input_path: Path,
        mode: str = "full",
        max_pages: int | None = None,
        workers: int = DEFAULT_DECODE_WORKERS,
        verbose: bool = False,
    ) -> RawSampleData:
        return load_geneactive_samples(
            input_path=input_path,
            mode=mode,
            max_pages=max_pages,
            workers=workers,
            verbose=verbose,
        )
