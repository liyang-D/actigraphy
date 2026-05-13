from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from models import EpochSummaryData, RawSampleData, coerce_full_sensor_columns
from utils import (
    ensure_csv_path,
    metadata_path_for_csv,
    normalize_epoch_label,
    parse_float,
    parse_timestamp,
)


def default_raw_metadata_path(input_csv_path: Path) -> Path:
    return metadata_path_for_csv(input_csv_path)


def default_epoch_output_csv_path(
    input_csv_path: Path,
    output_dir: Path | None,
    epoch: str,
) -> Path:
    input_csv_path = Path(input_csv_path)
    output_dir = input_csv_path.parent if output_dir is None else output_dir

    output_base = input_csv_path.stem
    if output_base.endswith("_raw"):
        output_base = output_base[:-4]

    epoch_label = normalize_epoch_label(epoch)
    return ensure_csv_path(
        output_dir / f"{output_base}_{epoch_label}.csv",
        "Output CSV path",
    )


def default_epoch_metadata_path(output_csv_path: Path) -> Path:
    return metadata_path_for_csv(output_csv_path)


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_sample_rate_hz(
    metadata: dict[str, Any],
    fallback_sample_rate_hz: float | None = None,
) -> float:
    recording = metadata.get("recording", {})
    raw_sample_rate = recording.get("measurement_frequency_hz")
    sample_rate = parse_float(str(raw_sample_rate)) if raw_sample_rate is not None else None

    if sample_rate is None:
        sample_rate = fallback_sample_rate_hz

    if sample_rate is None:
        raise ValueError(
            "Sample rate was not found in the paired Step 1A metadata. "
            "Use the metadata file next to the input CSV or pass --sample-rate."
        )

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    return sample_rate


def load_raw_sample_csv(
    csv_path: Path,
    fallback_sample_rate_hz: float | None = None,
    verbose: bool = False,
) -> RawSampleData:
    csv_path = ensure_csv_path(Path(csv_path), "Input CSV path")
    metadata_path = default_raw_metadata_path(csv_path)

    if verbose:
        print(f"Loading raw sample CSV: {csv_path}")
        if metadata_path.exists():
            print(f"Using paired metadata: {metadata_path}")
        else:
            print(f"Paired metadata not found: {metadata_path}")

    metadata = load_metadata(metadata_path) if metadata_path.exists() else {}
    sample_rate_hz = get_sample_rate_hz(
        metadata=metadata,
        fallback_sample_rate_hz=fallback_sample_rate_hz,
    )

    if verbose:
        print(f"Sample rate: {sample_rate_hz} Hz")

    data = pd.read_csv(csv_path)

    if "Time" not in data.columns:
        raise ValueError("Input CSV must contain a 'Time' column.")

    if verbose:
        print(f"Raw samples loaded: {len(data)} rows")
        print("Parsing timestamps and numeric columns")

    data["Time"] = pd.to_datetime(data["Time"].map(parse_timestamp))

    for column in ["Ax", "Ay", "Az"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = coerce_full_sensor_columns(data)

    data = data.sort_values("Time").reset_index(drop=True)

    raw_data = RawSampleData(
        data=data,
        metadata=metadata,
        sample_rate_hz=sample_rate_hz,
    )
    raw_data.validate()

    if verbose:
        print("Raw sample CSV ready for preprocessing")

    return raw_data


def save_epoch_summary_csv(summary_data: EpochSummaryData, output_path: Path) -> None:
    summary_data.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data.data.to_csv(output_path, index=False)
