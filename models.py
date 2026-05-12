from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


RAW_MOTION_COLUMNS = ["Time", "Ax", "Ay", "Az"]
RAW_FULL_COLUMNS = ["Time", "Ax", "Ay", "Az", "Lux", "Button", "Temperature"]

EPOCH_OUTPUT_COLUMNS: dict[str, list[str]] = {
    "svm": ["Time", "SVM_sum"],
    "motion-summary": [
        "Time",
        "Ax_mean",
        "Ay_mean",
        "Az_mean",
        "SVM_sum",
        "Ax_sd",
        "Ay_sd",
        "Az_sd",
    ],
    "full-summary": [
        "Time",
        "Ax_mean",
        "Ay_mean",
        "Az_mean",
        "Lux_mean",
        "Button_sum",
        "Temperature_mean",
        "SVM_sum",
        "Ax_sd",
        "Ay_sd",
        "Az_sd",
        "Lux_peak",
    ],
}


def get_epoch_output_columns(summary_mode: str) -> list[str]:
    try:
        return list(EPOCH_OUTPUT_COLUMNS[summary_mode])
    except KeyError as exc:
        raise ValueError(
            "summary_mode must be one of 'svm', 'motion-summary', or 'full-summary'."
        ) from exc


@dataclass
class RawSampleData:
    """Standard sample-level data passed from Step 1A to Step 1B."""

    data: pd.DataFrame
    metadata: dict[str, Any]
    sample_rate_hz: float

    def validate(self, require_full: bool = False) -> None:
        required_columns = RAW_FULL_COLUMNS if require_full else RAW_MOTION_COLUMNS
        missing = [column for column in required_columns if column not in self.data.columns]

        if missing:
            raise ValueError(f"Raw sample data is missing required columns: {missing}")

        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")

    def copy(self) -> "RawSampleData":
        return RawSampleData(
            data=self.data.copy(),
            metadata=dict(self.metadata),
            sample_rate_hz=self.sample_rate_hz,
        )


@dataclass
class EpochSummaryData:
    """Epoch-level Step 1B output."""

    data: pd.DataFrame
    metadata: dict[str, Any]
    summary_mode: str

    def validate(self) -> None:
        expected_columns = get_epoch_output_columns(self.summary_mode)
        missing = [column for column in expected_columns if column not in self.data.columns]

        if missing:
            raise ValueError(f"Epoch summary data is missing required columns: {missing}")

    def copy(self) -> "EpochSummaryData":
        return EpochSummaryData(
            data=self.data.copy(),
            metadata=dict(self.metadata),
            summary_mode=self.summary_mode,
        )


@dataclass
class PreprocessConfig:
    """Configuration for Step 1B preprocessing."""

    epoch: str = "60s"
    filter_enabled: bool = True
    filter_type: str = "butterworth"
    filter_mode: str = "bandpass"
    filter_order: int = 4
    low_cutoff_hz: float = 0.5
    high_cutoff_hz: float = 20.0
    summary_mode: str = "full-summary"
    svm_method: str = "geneactive_abs"
    time_label: str = "epoch_end"
    standard_deviation_ddof: int = 0

    def validate(self, sample_rate_hz: float) -> None:
        if not self.epoch:
            raise ValueError("epoch must be provided.")

        get_epoch_output_columns(self.summary_mode)

        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")

        if self.filter_enabled:
            if self.filter_type != "butterworth":
                raise ValueError("filter_type must be 'butterworth'.")

            if self.filter_mode != "bandpass":
                raise ValueError("filter_mode must be 'bandpass'.")

            if self.filter_order <= 0:
                raise ValueError("filter_order must be positive.")

            if self.low_cutoff_hz <= 0:
                raise ValueError("low_cutoff_hz must be positive.")

            if self.high_cutoff_hz <= 0:
                raise ValueError("high_cutoff_hz must be positive.")

            if self.low_cutoff_hz >= self.high_cutoff_hz:
                raise ValueError("low_cutoff_hz must be smaller than high_cutoff_hz.")

            nyquist = 0.5 * sample_rate_hz
            if self.high_cutoff_hz >= nyquist:
                raise ValueError(
                    "high_cutoff_hz must be smaller than the Nyquist frequency "
                    f"({nyquist} Hz)."
                )

        if self.svm_method != "geneactive_abs":
            raise ValueError("svm_method must be 'geneactive_abs'.")

        if self.time_label != "epoch_end":
            raise ValueError("time_label must be 'epoch_end'.")

        if self.standard_deviation_ddof < 0:
            raise ValueError("standard_deviation_ddof must be non-negative.")
