from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from models import EpochSummaryData, PreprocessConfig, RawSampleData, get_epoch_output_columns
from preprocess.epoch import aggregate_epochs
from preprocess.filter import apply_butterworth_bandpass
from preprocess.io import (
    default_epoch_metadata_path,
    default_epoch_output_csv_path,
    load_raw_sample_csv,
    save_epoch_summary_csv,
    save_metadata,
)
from preprocess.svm import add_geneactive_svm
from utils import ensure_csv_path, validate_output_precision


def update_metadata_for_preprocessing(
    metadata: dict[str, Any],
    input_csv_path: Path,
    output_csv_path: Path,
    config: PreprocessConfig,
    sample_rate_hz: float,
) -> dict[str, Any]:
    metadata = copy.deepcopy(metadata)

    metadata["preprocessing"] = {
        "input_file": str(input_csv_path),
        "sample_rate_hz": sample_rate_hz,
        "epoch": config.epoch,
        "summary_mode": config.summary_mode,
        "svm_method": config.svm_method,
        "time_label": config.time_label,
        "standard_deviation_ddof": config.standard_deviation_ddof,
        "filter": {
            "enabled": config.filter_enabled,
            "type": config.filter_type,
            "mode": config.filter_mode,
            "order": config.filter_order,
            "low_cutoff_hz": config.low_cutoff_hz,
            "high_cutoff_hz": config.high_cutoff_hz,
            "applied_columns": ["Ax", "Ay", "Az"],
        },
    }
    metadata["preprocess_output"] = {
        "output_file": str(output_csv_path),
        "columns": get_epoch_output_columns(config.summary_mode),
    }

    return metadata


def run_preprocessing(
    raw_data: RawSampleData,
    config: PreprocessConfig,
    verbose: bool = False,
) -> EpochSummaryData:
    require_full = config.summary_mode == "full-summary"
    raw_data.validate(require_full=require_full)
    config.validate(sample_rate_hz=raw_data.sample_rate_hz)

    if verbose:
        print(f"Preprocessing with mode: {config.summary_mode}")
        print(f"Epoch length: {config.epoch}")
        print(f"Input samples: {len(raw_data.data)}")
        print(f"Sample rate: {raw_data.sample_rate_hz} Hz")

    working_data = raw_data.data.copy()

    if config.filter_enabled:
        if verbose:
            print(
                "Applying Butterworth bandpass filter "
                f"({config.low_cutoff_hz}-{config.high_cutoff_hz} Hz)"
            )

        working_data = apply_butterworth_bandpass(
            data=working_data,
            sample_rate_hz=raw_data.sample_rate_hz,
            low_cutoff_hz=config.low_cutoff_hz,
            high_cutoff_hz=config.high_cutoff_hz,
            order=config.filter_order,
        )
    elif verbose:
        print("Filtering disabled")

    if verbose:
        print("Computing SVM")
    working_data = add_geneactive_svm(working_data)

    if verbose:
        print("Aggregating epochs")
    summary = aggregate_epochs(
        data=working_data,
        epoch=config.epoch,
        summary_mode=config.summary_mode,
        standard_deviation_ddof=config.standard_deviation_ddof,
        anchor_time=raw_data.data["Time"].iloc[0],
    )

    if verbose:
        print(f"Epoch rows generated: {len(summary)}")

    return EpochSummaryData(
        data=summary,
        metadata=raw_data.metadata,
        summary_mode=config.summary_mode,
    )


def preprocess_file(
    input_csv_path: Path,
    output_csv_path: Path | None = None,
    output_dir: Path | None = None,
    config: PreprocessConfig | None = None,
    fallback_sample_rate_hz: float | None = None,
    precision: int = 5,
    verbose: bool = False,
) -> tuple[Path, Path]:
    input_csv_path = Path(input_csv_path)
    config = PreprocessConfig() if config is None else config
    precision = validate_output_precision(precision)
    if output_csv_path is not None:
        output_csv_path = ensure_csv_path(output_csv_path, "Output CSV path")

    raw_data = load_raw_sample_csv(
        csv_path=input_csv_path,
        fallback_sample_rate_hz=fallback_sample_rate_hz,
        verbose=verbose,
    )

    return preprocess_raw_data_to_files(
        raw_data=raw_data,
        input_path=input_csv_path,
        output_csv_path=output_csv_path,
        output_dir=output_dir,
        config=config,
        precision=precision,
        verbose=verbose,
    )


def preprocess_raw_data_to_files(
    raw_data: RawSampleData,
    input_path: Path,
    output_csv_path: Path | None = None,
    output_dir: Path | None = None,
    config: PreprocessConfig | None = None,
    precision: int = 5,
    verbose: bool = False,
) -> tuple[Path, Path]:
    input_path = Path(input_path)
    config = PreprocessConfig() if config is None else config
    precision = validate_output_precision(precision)

    if output_csv_path is None:
        output_csv_path = default_epoch_output_csv_path(
            input_csv_path=input_path,
            output_dir=output_dir,
            epoch=config.epoch,
        )
    else:
        output_csv_path = ensure_csv_path(output_csv_path, "Output CSV path")

    output_metadata_path = default_epoch_metadata_path(output_csv_path)

    summary_data = run_preprocessing(
        raw_data=raw_data,
        config=config,
        verbose=verbose,
    )

    metadata = update_metadata_for_preprocessing(
        metadata=summary_data.metadata,
        input_csv_path=input_path,
        output_csv_path=output_csv_path,
        config=config,
        sample_rate_hz=raw_data.sample_rate_hz,
    )
    summary_data.metadata = metadata

    if verbose:
        print(f"Writing epoch CSV: {output_csv_path}")
    save_epoch_summary_csv(summary_data, output_csv_path, precision=precision)

    if verbose:
        print(f"Writing metadata: {output_metadata_path}")
    save_metadata(metadata, output_metadata_path)

    if verbose:
        print(f"CSV saved to: {output_csv_path}")
        print(f"Metadata saved to: {output_metadata_path}")

    return output_csv_path, output_metadata_path
