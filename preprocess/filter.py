from __future__ import annotations

import pandas as pd
from scipy.signal import butter, lfilter


AXIS_COLUMNS = ["Ax", "Ay", "Az"]


def butter_bandpass(
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
):
    nyquist = 0.5 * sample_rate_hz
    normal_low = low_cutoff_hz / nyquist
    normal_high = high_cutoff_hz / nyquist

    return butter(
        order,
        [normal_low, normal_high],
        btype="bandpass",
        analog=False,
    )


def apply_butterworth_bandpass(
    data: pd.DataFrame,
    sample_rate_hz: float,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    order: int = 4,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = AXIS_COLUMNS if columns is None else columns
    filtered = data.copy()
    b, a = butter_bandpass(
        low_cutoff_hz=low_cutoff_hz,
        high_cutoff_hz=high_cutoff_hz,
        sample_rate_hz=sample_rate_hz,
        order=order,
    )

    for column in columns:
        filtered[column] = lfilter(
            b,
            a,
            pd.to_numeric(filtered[column], errors="coerce").to_numpy(dtype=float),
        )

    return filtered
