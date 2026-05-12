# readers/geneactive/pages.py

from __future__ import annotations

from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Iterator

from .models import GeneActivePage, GeneActivePageHeader
from utils import (
    parse_first_float,
    parse_float,
    parse_int,
    parse_key_value_lines,
    split_key_value,
)


PAGE_BLOCK_SIZE = 10


def parse_geneactive_datetime(value: str | None) -> datetime:
    if not value:
        raise ValueError("Missing GENEActiv datetime value.")

    value = value.strip()

    for fmt in (
        "%Y-%m-%d %H:%M:%S:%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported GENEActiv datetime format: {value}")


def parse_measurement_frequency(value: str | None) -> float:
    if value is None:
        raise ValueError("Missing measurement frequency.")

    parsed = parse_first_float(value)

    if parsed is None:
        raise ValueError(f"Invalid measurement frequency: {value}")

    return parsed


def validate_page_block(block_lines: list[str], page_number: int | None = None) -> None:
    prefix = f"Page {page_number}: " if page_number is not None else ""

    if len(block_lines) != PAGE_BLOCK_SIZE:
        raise ValueError(
            f"{prefix}expected {PAGE_BLOCK_SIZE} lines, got {len(block_lines)}."
        )

    if block_lines[0].strip() != "Recorded Data":
        raise ValueError(
            f"{prefix}expected 'Recorded Data', got {block_lines[0]!r}."
        )

    expected_keys = [
        "Device Unique Serial Code",
        "Sequence Number",
        "Page Time",
        "Unassigned",
        "Temperature",
        "Battery voltage",
        "Device Status",
        "Measurement Frequency",
    ]

    for offset, expected_key in enumerate(expected_keys, start=1):
        actual_key, _ = split_key_value(block_lines[offset].strip())

        if actual_key != expected_key:
            raise ValueError(
                f"{prefix}expected key {expected_key!r} on block line {offset}, "
                f"got {actual_key!r}."
            )

    if not block_lines[9].strip():
        raise ValueError(f"{prefix}empty hex data line.")


def parse_page_block(block_lines: list[str]) -> GeneActivePage:
    validate_page_block(block_lines)

    fields = parse_key_value_lines(block_lines[1:9])

    sequence_number = parse_int(fields.get("Sequence Number"))
    if sequence_number is None:
        raise ValueError("Invalid or missing sequence number.")

    page_time = parse_geneactive_datetime(fields.get("Page Time"))

    temperature = parse_float(fields.get("Temperature"))
    battery_voltage = parse_float(fields.get("Battery voltage"))
    device_status = fields.get("Device Status")
    measurement_frequency_hz = parse_measurement_frequency(
        fields.get("Measurement Frequency")
    )

    page_header = GeneActivePageHeader(
        sequence_number=sequence_number,
        page_time=page_time,
        temperature=temperature,
        battery_voltage=battery_voltage,
        device_status=device_status,
        measurement_frequency_hz=measurement_frequency_hz,
    )

    return GeneActivePage(
        header=page_header,
        hex_data=block_lines[9].strip(),
    )


def iter_page_blocks(
    path: Path,
    start_line_index: int = 59,
    max_pages: int | None = None,
) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(start_line_index):
            next(f, None)

        page_count = 0

        while True:
            if max_pages is not None and page_count >= max_pages:
                break

            block = [line.rstrip("\n\r") for line in islice(f, PAGE_BLOCK_SIZE)]

            if not block:
                break

            if len(block) < PAGE_BLOCK_SIZE:
                raise ValueError(
                    f"Incomplete page block at page index {page_count}: "
                    f"expected {PAGE_BLOCK_SIZE} lines, got {len(block)}."
                )

            yield block
            page_count += 1


def iter_geneactive_pages(
    path: Path,
    start_line_index: int = 59,
    max_pages: int | None = None,
) -> Iterator[GeneActivePage]:
    for page_index, block in enumerate(
        iter_page_blocks(
            path=path,
            start_line_index=start_line_index,
            max_pages=max_pages,
        )
    ):
        try:
            yield parse_page_block(block)
        except Exception as exc:
            raise ValueError(f"Failed to parse page block {page_index}.") from exc
