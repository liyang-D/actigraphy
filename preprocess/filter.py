# preprocess/filter.py

import numpy as np
from scipy.signal import butter, lfilter

from models import RawTriAxialData


def butter_bandpass(lo_cutoff: float, hi_cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_lo_cutoff = lo_cutoff / nyq
    normal_hi_cutoff = hi_cutoff / nyq
    b, a = butter(
        order,
        [normal_lo_cutoff, normal_hi_cutoff],
        btype="bandpass",
        analog=False,
    )
    return b, a


def butter_bandpass_filter(
    data,
    lo_cutoff: float,
    hi_cutoff: float,
    fs: float,
    order: int = 4,
):
    b, a = butter_bandpass(lo_cutoff, hi_cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_bandpass_filter(
    raw_data: RawTriAxialData,
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: float,
    order: int = 4,
) -> RawTriAxialData:
    raw_data.validate()

    filtered = raw_data.copy()
    filtered.data["Ax"] = butter_bandpass_filter(
        filtered.data["Ax"], low_cutoff, high_cutoff, sample_rate, order=order
    )
    filtered.data["Ay"] = butter_bandpass_filter(
        filtered.data["Ay"], low_cutoff, high_cutoff, sample_rate, order=order
    )
    filtered.data["Az"] = butter_bandpass_filter(
        filtered.data["Az"], low_cutoff, high_cutoff, sample_rate, order=order
    )
    return filtered