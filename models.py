from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class RawTriAxialData:
    """
    Unified raw tri-axial time series passed from Step 1a to Step 1b.

    Required columns:
        Time, Ax, Ay, Az
    """

    data: pd.DataFrame
    sample_rate: Optional[float] = None

    def validate(self) -> None:
        required_columns = ["Time", "Ax", "Ay", "Az"]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"RawTriAxialData is missing required columns: {missing}"
            )

    def copy(self) -> "RawTriAxialData":
        return RawTriAxialData(
            data=self.data.copy(),
            sample_rate=self.sample_rate,
        )


@dataclass
class EpochSVMData:
    """
    Output of Step 1b after SVM computation and epoching.

    Expected column:
        SVM

    The time information is expected to be carried by the index
    or by a Time column, depending on the export stage.
    """

    data: pd.DataFrame
    binsize: str

    def validate(self) -> None:
        if "SVM" not in self.data.columns:
            raise ValueError("EpochSVMData must contain an 'SVM' column.")

    def copy(self) -> "EpochSVMData":
        return EpochSVMData(
            data=self.data.copy(),
            binsize=self.binsize,
        )


@dataclass
class PreprocessConfig:
    """
    Configuration for Step 1b preprocessing.
    """

    binsize: str = "3T"
    apply_filter: bool = True
    sample_rate: float = 100.0
    low_cutoff: float = 0.5
    high_cutoff: float = 20.0

    def validate(self) -> None:
        if not self.binsize:
            raise ValueError("binsize must be provided.")

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if self.low_cutoff <= 0:
            raise ValueError("low_cutoff must be positive.")

        if self.high_cutoff <= 0:
            raise ValueError("high_cutoff must be positive.")

        if self.low_cutoff >= self.high_cutoff:
            raise ValueError("low_cutoff must be smaller than high_cutoff.")

        nyquist = 0.5 * self.sample_rate
        if self.high_cutoff >= nyquist:
            raise ValueError(
                "high_cutoff must be smaller than the Nyquist frequency "
                f"({nyquist} Hz)."
            )