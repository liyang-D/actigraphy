from __future__ import annotations

import pandas as pd

from models import get_epoch_output_columns
from preprocess.svm import SVM_SAMPLE_COLUMN
from utils import format_timestamp_millis


EPOCH_TIME_COLUMN = "_epoch_time"


def add_epoch_time(
    data: pd.DataFrame,
    epoch: str,
    anchor_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    data = data.copy()
    anchor_time = data["Time"].iloc[0] if anchor_time is None else anchor_time
    epoch_offset = pd.to_timedelta(epoch)
    elapsed = data["Time"] - anchor_time
    epoch_index = elapsed // epoch_offset
    data[EPOCH_TIME_COLUMN] = anchor_time + (epoch_index * epoch_offset)
    return data


def aggregate_epochs(
    data: pd.DataFrame,
    epoch: str,
    summary_mode: str,
    standard_deviation_ddof: int = 0,
    anchor_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot aggregate an empty input CSV.")

    data = add_epoch_time(data, epoch=epoch, anchor_time=anchor_time)
    grouped = data.groupby(EPOCH_TIME_COLUMN, sort=True)

    output = pd.DataFrame(index=grouped.size().index)
    output["Time"] = [
        format_timestamp_millis(value.to_pydatetime())
        for value in output.index
    ]
    output["Ax_mean"] = grouped["Ax"].mean().to_numpy()
    output["Ay_mean"] = grouped["Ay"].mean().to_numpy()
    output["Az_mean"] = grouped["Az"].mean().to_numpy()
    output["SVM_sum"] = grouped[SVM_SAMPLE_COLUMN].sum().to_numpy()
    output["Ax_sd"] = grouped["Ax"].std(ddof=standard_deviation_ddof).fillna(0).to_numpy()
    output["Ay_sd"] = grouped["Ay"].std(ddof=standard_deviation_ddof).fillna(0).to_numpy()
    output["Az_sd"] = grouped["Az"].std(ddof=standard_deviation_ddof).fillna(0).to_numpy()

    if summary_mode == "full-summary":
        output["Lux_mean"] = grouped["Lux"].mean().to_numpy()
        output["Button_sum"] = grouped["Button"].sum().to_numpy()
        output["Temperature_mean"] = grouped["Temperature"].mean().to_numpy()
        output["Lux_peak"] = grouped["Lux"].max().to_numpy()

    columns = get_epoch_output_columns(summary_mode)
    return output.reset_index(drop=True)[columns]
