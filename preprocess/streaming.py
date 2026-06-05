from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from models import EpochSummaryData, PreprocessConfig, get_epoch_output_columns
from preprocess.filter import AXIS_COLUMNS, butter_bandpass
from preprocess.io import (
    default_epoch_metadata_path,
    default_epoch_output_csv_path,
    default_raw_metadata_path,
    get_sample_rate_hz,
    load_metadata,
    save_epoch_summary_csv,
    save_metadata,
)
from preprocess.pipeline import update_metadata_for_preprocessing
from utils import ensure_csv_path, format_timestamp_millis, parse_timestamp, validate_output_precision


DEFAULT_STREAM_CHUNK_ROWS = 500_000


@dataclass
class AxisStats:
    sum: float = 0.0
    sum_sq: float = 0.0
    count: int = 0

    def update(self, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        self.sum += float(values.sum(dtype=np.float64))
        self.sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        self.count += int(values.size)

    def mean(self) -> float:
        if self.count == 0:
            return math.nan
        return self.sum / self.count

    def std(self, ddof: int) -> float:
        if self.count - ddof <= 0:
            return 0.0

        variance = (self.sum_sq - (self.sum * self.sum / self.count)) / (self.count - ddof)
        return math.sqrt(max(variance, 0.0))


@dataclass
class EpochStats:
    axes: dict[str, AxisStats] = field(
        default_factory=lambda: {column: AxisStats() for column in AXIS_COLUMNS}
    )
    svm_sum: float = 0.0
    lux_sum: float = 0.0
    lux_count: int = 0
    lux_peak: float = math.nan
    button_sum: float = 0.0
    temperature_sum: float = 0.0
    temperature_count: int = 0

    def update_numeric_mean(self, values: np.ndarray, name: str) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        if name == "Lux":
            self.lux_sum += float(values.sum(dtype=np.float64))
            self.lux_count += int(values.size)
            peak = float(values.max())
            self.lux_peak = peak if math.isnan(self.lux_peak) else max(self.lux_peak, peak)
            return

        if name == "Temperature":
            self.temperature_sum += float(values.sum(dtype=np.float64))
            self.temperature_count += int(values.size)

    def lux_mean(self) -> float:
        if self.lux_count == 0:
            return math.nan
        return self.lux_sum / self.lux_count

    def temperature_mean(self) -> float:
        if self.temperature_count == 0:
            return math.nan
        return self.temperature_sum / self.temperature_count


class StreamingButterworthBandpass:
    def __init__(
        self,
        sample_rate_hz: float,
        low_cutoff_hz: float,
        high_cutoff_hz: float,
        order: int,
        columns: Iterable[str] = AXIS_COLUMNS,
    ) -> None:
        self.b, self.a = butter_bandpass(
            low_cutoff_hz=low_cutoff_hz,
            high_cutoff_hz=high_cutoff_hz,
            sample_rate_hz=sample_rate_hz,
            order=order,
        )
        state_length = max(len(self.a), len(self.b)) - 1
        self.states = {
            column: np.zeros(state_length, dtype=np.float64)
            for column in columns
        }

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        filtered = data.copy()

        for column, state in self.states.items():
            values = pd.to_numeric(filtered[column], errors="coerce").to_numpy(dtype=np.float64)
            filtered_values, next_state = lfilter(self.b, self.a, values, zi=state)
            filtered[column] = filtered_values
            self.states[column] = next_state

        return filtered


class StreamingEpochAccumulator:
    def __init__(
        self,
        epoch: str,
        summary_mode: str,
        standard_deviation_ddof: int = 0,
    ) -> None:
        self.epoch_offset = pd.to_timedelta(epoch)
        self.summary_mode = summary_mode
        self.standard_deviation_ddof = standard_deviation_ddof
        self.anchor_time: pd.Timestamp | None = None
        self.last_time: pd.Timestamp | None = None
        self.epochs: dict[pd.Timestamp, EpochStats] = {}

    def update(self, data: pd.DataFrame) -> None:
        if data.empty:
            return

        data = data.copy()
        if self.anchor_time is None:
            self.anchor_time = data["Time"].iloc[0]

        if self.last_time is not None and data["Time"].iloc[0] < self.last_time:
            raise ValueError(
                "Streaming preprocessing requires input samples sorted by Time. "
                "Run the existing pandas preprocessing path for unsorted CSV files."
            )

        if not data["Time"].is_monotonic_increasing:
            raise ValueError(
                "Streaming preprocessing requires input samples sorted by Time. "
                "Run the existing pandas preprocessing path for unsorted CSV files."
            )

        self.last_time = data["Time"].iloc[-1]

        elapsed = data["Time"] - self.anchor_time
        epoch_index = elapsed // self.epoch_offset
        data["_epoch_time"] = self.anchor_time + (epoch_index * self.epoch_offset)

        axes = data[AXIS_COLUMNS].apply(pd.to_numeric, errors="coerce")
        vector_magnitude = np.sqrt((axes.to_numpy(dtype=np.float64) ** 2).sum(axis=1))
        data["_svm_sample"] = np.abs(vector_magnitude - 1.0)

        for epoch_time, group in data.groupby("_epoch_time", sort=False):
            stats = self.epochs.setdefault(epoch_time, EpochStats())

            for column in AXIS_COLUMNS:
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
                stats.axes[column].update(values)

            svm_values = pd.to_numeric(group["_svm_sample"], errors="coerce").to_numpy(dtype=np.float64)
            svm_values = svm_values[np.isfinite(svm_values)]
            if svm_values.size:
                stats.svm_sum += float(svm_values.sum(dtype=np.float64))

            if self.summary_mode == "full-summary":
                lux_values = pd.to_numeric(group["Lux"], errors="coerce").to_numpy(dtype=np.float64)
                stats.update_numeric_mean(lux_values, "Lux")

                button_values = pd.to_numeric(group["Button"], errors="coerce").to_numpy(dtype=np.float64)
                button_values = button_values[np.isfinite(button_values)]
                if button_values.size:
                    stats.button_sum += float(button_values.sum(dtype=np.float64))

                temperature_values = pd.to_numeric(group["Temperature"], errors="coerce").to_numpy(dtype=np.float64)
                stats.update_numeric_mean(temperature_values, "Temperature")

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for epoch_time in sorted(self.epochs):
            stats = self.epochs[epoch_time]
            row: dict[str, Any] = {
                "Time": format_timestamp_millis(epoch_time.to_pydatetime()),
                "Ax_mean": stats.axes["Ax"].mean(),
                "Ay_mean": stats.axes["Ay"].mean(),
                "Az_mean": stats.axes["Az"].mean(),
                "SVM_sum": stats.svm_sum,
                "Ax_sd": stats.axes["Ax"].std(self.standard_deviation_ddof),
                "Ay_sd": stats.axes["Ay"].std(self.standard_deviation_ddof),
                "Az_sd": stats.axes["Az"].std(self.standard_deviation_ddof),
            }

            if self.summary_mode == "full-summary":
                row.update(
                    {
                        "Lux_mean": stats.lux_mean(),
                        "Button_sum": stats.button_sum,
                        "Temperature_mean": stats.temperature_mean(),
                        "Lux_peak": stats.lux_peak,
                    }
                )

            rows.append(row)

        columns = get_epoch_output_columns(self.summary_mode)
        return pd.DataFrame(rows, columns=columns)


def prepare_raw_chunk(data: pd.DataFrame, require_full: bool) -> pd.DataFrame:
    required_columns = ["Time", "Ax", "Ay", "Az"]
    if require_full:
        required_columns.extend(["Lux", "Button", "Temperature"])

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Raw sample data is missing required columns: {missing}")

    data = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["Time"]):
        data["Time"] = pd.to_datetime(
            data["Time"].map(
                lambda value: value
                if isinstance(value, pd.Timestamp)
                else parse_timestamp(str(value))
            )
        )

    for column in AXIS_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if require_full:
        data["Lux"] = pd.to_numeric(data["Lux"], errors="coerce")
        if data["Lux"].isna().any():
            raise ValueError("Lux column must contain numeric values.")
        data["Button"] = pd.to_numeric(data["Button"], errors="coerce")
        if data["Button"].isna().any() or not data["Button"].isin([0, 1]).all():
            raise ValueError("Button column must contain only 0/1 values.")
        data["Temperature"] = pd.to_numeric(data["Temperature"], errors="coerce")
        if data["Temperature"].isna().any():
            raise ValueError("Temperature column must contain numeric values.")

    return data


def iter_raw_csv_chunks(
    csv_path: Path,
    require_full: bool,
    chunk_rows: int = DEFAULT_STREAM_CHUNK_ROWS,
) -> Iterator[pd.DataFrame]:
    csv_path = ensure_csv_path(Path(csv_path), "Input CSV path")

    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        yield prepare_raw_chunk(chunk, require_full=require_full)


def run_streaming_preprocessing(
    chunks: Iterable[pd.DataFrame],
    metadata: dict[str, Any],
    sample_rate_hz: float,
    config: PreprocessConfig,
    verbose: bool = False,
) -> EpochSummaryData:
    require_full = config.summary_mode == "full-summary"
    config.validate(sample_rate_hz=sample_rate_hz)

    if verbose:
        print(f"Streaming preprocessing with mode: {config.summary_mode}")
        print(f"Epoch length: {config.epoch}")
        print(f"Sample rate: {sample_rate_hz} Hz")

    accumulator = StreamingEpochAccumulator(
        epoch=config.epoch,
        summary_mode=config.summary_mode,
        standard_deviation_ddof=config.standard_deviation_ddof,
    )
    stream_filter: StreamingButterworthBandpass | None = None
    if config.filter_enabled:
        stream_filter = StreamingButterworthBandpass(
            sample_rate_hz=sample_rate_hz,
            low_cutoff_hz=config.low_cutoff_hz,
            high_cutoff_hz=config.high_cutoff_hz,
            order=config.filter_order,
        )

    chunk_count = 0
    sample_count = 0
    for chunk in chunks:
        chunk = prepare_raw_chunk(chunk, require_full=require_full)
        if stream_filter is not None:
            chunk = stream_filter.apply(chunk)

        accumulator.update(chunk)
        chunk_count += 1
        sample_count += len(chunk)

        if verbose:
            print(f"Processed streaming chunk {chunk_count}: {len(chunk)} samples")

    summary = accumulator.to_dataframe()

    if summary.empty:
        raise ValueError("Cannot aggregate an empty input CSV.")

    if verbose:
        print(f"Streaming samples processed: {sample_count}")
        print(f"Epoch rows generated: {len(summary)}")

    return EpochSummaryData(
        data=summary,
        metadata=metadata,
        summary_mode=config.summary_mode,
    )


def preprocess_file_streaming(
    input_csv_path: Path,
    output_csv_path: Path | None = None,
    output_dir: Path | None = None,
    config: PreprocessConfig | None = None,
    fallback_sample_rate_hz: float | None = None,
    precision: int = 5,
    chunk_rows: int = DEFAULT_STREAM_CHUNK_ROWS,
    verbose: bool = False,
) -> tuple[Path, Path]:
    input_csv_path = ensure_csv_path(Path(input_csv_path), "Input CSV path")
    config = PreprocessConfig() if config is None else config
    precision = validate_output_precision(precision)

    metadata_path = default_raw_metadata_path(input_csv_path)
    metadata = load_metadata(metadata_path) if metadata_path.exists() else {}
    sample_rate_hz = get_sample_rate_hz(
        metadata=metadata,
        fallback_sample_rate_hz=fallback_sample_rate_hz,
    )

    if output_csv_path is None:
        output_csv_path = default_epoch_output_csv_path(
            input_csv_path=input_csv_path,
            output_dir=output_dir,
            epoch=config.epoch,
        )
    else:
        output_csv_path = ensure_csv_path(output_csv_path, "Output CSV path")

    return preprocess_chunks_to_files(
        chunks=iter_raw_csv_chunks(
            csv_path=input_csv_path,
            require_full=config.summary_mode == "full-summary",
            chunk_rows=chunk_rows,
        ),
        metadata=metadata,
        sample_rate_hz=sample_rate_hz,
        input_path=input_csv_path,
        output_csv_path=output_csv_path,
        output_dir=output_dir,
        config=config,
        precision=precision,
        verbose=verbose,
        streaming_chunk_rows=chunk_rows,
    )


def preprocess_chunks_to_files(
    chunks: Iterable[pd.DataFrame],
    metadata: dict[str, Any],
    sample_rate_hz: float,
    input_path: Path,
    output_csv_path: Path | None = None,
    output_dir: Path | None = None,
    config: PreprocessConfig | None = None,
    precision: int = 5,
    verbose: bool = False,
    streaming_chunk_rows: int | None = None,
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
    summary_data = run_streaming_preprocessing(
        chunks=chunks,
        metadata=metadata,
        sample_rate_hz=sample_rate_hz,
        config=config,
        verbose=verbose,
    )
    metadata = update_metadata_for_preprocessing(
        metadata=summary_data.metadata,
        input_csv_path=input_path,
        output_csv_path=output_csv_path,
        config=config,
        sample_rate_hz=sample_rate_hz,
    )
    metadata["preprocessing"]["streaming"] = {
        "enabled": True,
        "chunk_rows": streaming_chunk_rows,
    }
    summary_data.metadata = metadata

    if verbose:
        print(f"Writing epoch CSV: {output_csv_path}")
    save_epoch_summary_csv(summary_data, output_csv_path, precision=precision)

    if verbose:
        print(f"Writing metadata: {output_metadata_path}")
    save_metadata(metadata, output_metadata_path)

    return output_csv_path, output_metadata_path
