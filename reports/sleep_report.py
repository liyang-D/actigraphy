from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "actigraphy_matplotlib"),
)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import parse_timestamp


DEFAULT_TITLE = "Actigraphy Sleep Report"
DEFAULT_DAY_START_HOUR = 15
DEFAULT_DAYS_PER_PAGE = 4
DEFAULT_ACTIVITY_SCALE_PERCENTILE = 99.0
DEFAULT_LUX_LOG_SCALE_MAX = 5.0
ACTIVITY_COLOR = "#292b55"
LIGHT_COLOR = "#ff8a3d"
GRID_COLOR = "#eeeeee"


def ensure_pdf_path(path: Path, name: str = "Output PDF path") -> Path:
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"{name} must end with .pdf: {path}")

    return path


def default_report_output_path(input_csv_path: Path) -> Path:
    return ensure_pdf_path(input_csv_path.with_name(f"{input_csv_path.stem}_sleep_report.pdf"))


def report_date_text(report_date: str | None = None) -> str:
    if report_date:
        return report_date

    return datetime.now().strftime("%d %b %Y")


def find_first_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in data.columns:
            return column

    return None


def load_actigraphy_csv(input_csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_csv_path)

    if "Time" not in data.columns:
        raise ValueError("Input CSV must contain a 'Time' column.")

    data["Time"] = pd.to_datetime(data["Time"].map(parse_report_timestamp))
    data = data.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    if data.empty:
        raise ValueError("Input CSV does not contain any valid timestamped rows.")

    if "SVM_sum" not in data.columns:
        axis_columns = ["Ax", "Ay", "Az"]
        if not all(column in data.columns for column in axis_columns):
            raise ValueError(
                "Input CSV must contain either 'SVM_sum' or raw 'Ax', 'Ay', 'Az' columns."
            )

        axes = data[axis_columns].apply(pd.to_numeric, errors="coerce")
        data["SVM_sum"] = np.abs(np.sqrt((axes**2).sum(axis=1)) - 1)
    else:
        data["SVM_sum"] = pd.to_numeric(data["SVM_sum"], errors="coerce")

    for column in [
        "Lux_mean",
        "Lux_peak",
        "Lux",
        "Temperature_mean",
        "Temperature",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def resolve_activity_scale(data: pd.DataFrame, activity_scale: float | None) -> float:
    if activity_scale is not None:
        if activity_scale <= 0:
            raise ValueError("--activity-scale must be positive.")
        return activity_scale

    activity = data["SVM_sum"].dropna().clip(lower=0)
    if activity.empty:
        return 1.0

    scale = float(np.nanpercentile(activity, DEFAULT_ACTIVITY_SCALE_PERCENTILE))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(activity.max())

    return scale if scale > 0 else 1.0


def resolve_lux_log_scale(lux_log_scale_max: float) -> float:
    if lux_log_scale_max <= 0:
        raise ValueError("--lux-log-scale-max must be positive.")

    return lux_log_scale_max


def parse_report_timestamp(value: Any) -> datetime | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    try:
        return parse_timestamp(str(value))
    except ValueError:
        return pd.NaT


def day_window_starts(data: pd.DataFrame, day_start_hour: int) -> list[pd.Timestamp]:
    if day_start_hour < 0 or day_start_hour > 23:
        raise ValueError("--day-start-hour must be between 0 and 23.")

    offset = pd.to_timedelta(day_start_hour, unit="h")
    starts = (data["Time"] - offset).dt.floor("D") + offset
    return [pd.Timestamp(value) for value in sorted(starts.dropna().unique())]


def select_day_data(
    data: pd.DataFrame,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    window_end = window_start + pd.Timedelta(days=1)
    return data[(data["Time"] >= window_start) & (data["Time"] < window_end)]


def infer_bar_width_days(day_data: pd.DataFrame) -> float:
    if len(day_data) < 2:
        return 1 / (24 * 60)

    times = mdates.date2num(day_data["Time"])
    deltas = np.diff(times)
    positive_deltas = deltas[deltas > 0]

    if len(positive_deltas) == 0:
        return 1 / (24 * 60)

    return float(np.median(positive_deltas) * 0.8)


def day_label(window_start: pd.Timestamp, day_number: int) -> str:
    return f"{window_start.strftime('%A')}\n{window_start.strftime('%d %b %y')}\nDay {day_number}"


def style_time_axis(ax, window_start: pd.Timestamp) -> None:
    window_end = window_start + pd.Timedelta(days=1)
    ax.set_xlim(window_start, window_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.tick_params(axis="x", labelsize=8, colors="#444444")


def plot_day_panel(
    ax,
    data: pd.DataFrame,
    window_start: pd.Timestamp,
    day_number: int,
    show_legend: bool,
    activity_scale: float,
    lux_log_scale_max: float,
) -> None:
    day_data = select_day_data(data, window_start=window_start)
    style_time_axis(ax, window_start=window_start)
    ax.set_ylabel(
        day_label(window_start, day_number),
        rotation=90,
        labelpad=34,
        fontsize=9,
        color="#333333",
        va="center",
    )

    if day_data.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#777777",
        )
        return

    svm = day_data["SVM_sum"].fillna(0).clip(lower=0)
    ax.bar(
        day_data["Time"],
        svm,
        width=infer_bar_width_days(day_data),
        color=ACTIVITY_COLOR,
        alpha=0.85,
        linewidth=0,
    )
    ax.set_ylim(0, activity_scale)

    light_column = find_first_column(day_data, ["Lux_mean", "Lux_peak", "Lux"])
    light_plotted = False
    if light_column is not None and day_data[light_column].notna().any():
        light_ax = ax.twinx()
        lux = np.log10(day_data[light_column].fillna(0).clip(lower=0) + 1)
        light_ax.plot(
            day_data["Time"],
            lux,
            color=LIGHT_COLOR,
            linewidth=0.8,
            alpha=0.95,
        )
        light_ax.set_ylim(0, lux_log_scale_max)
        light_ax.tick_params(axis="y", length=0, labelright=False)
        for spine in light_ax.spines.values():
            spine.set_visible(False)
        light_plotted = True

    if show_legend:
        handles: list[Any] = [
            Patch(facecolor=ACTIVITY_COLOR, alpha=0.85, label="SVM activity"),
        ]

        if light_plotted:
            handles.append(
                Line2D([0], [0], color=LIGHT_COLOR, label=f"log10({light_column}+1)")
            )

        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=7,
            ncol=len(handles),
        )


def render_title_page(
    pdf: PdfPages,
    title: str,
    report_date: str,
    input_csv_path: Path,
    day_count: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.68, title, ha="center", va="center", fontsize=26, weight="bold")
    fig.text(
        0.5,
        0.58,
        f"Report Date  {report_date}",
        ha="center",
        va="center",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.48,
        f"Input CSV: {input_csv_path}",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    fig.text(
        0.5,
        0.43,
        f"Daily actigraphy windows: {day_count}",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    fig.text(0.95, 0.05, f"Page 1 of {total_pages}", ha="right", fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def render_day_page(
    pdf: PdfPages,
    data: pd.DataFrame,
    starts: list[pd.Timestamp],
    page_index: int,
    total_pages: int,
    title: str,
    report_date: str,
    first_day_index: int,
    activity_scale: float,
    lux_log_scale_max: float,
) -> None:
    fig, axes = plt.subplots(
        len(starts),
        1,
        figsize=(11.69, 8.27),
        sharex=False,
        constrained_layout=False,
    )
    if len(starts) == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.08, hspace=0.48)
    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=18, weight="bold")
    fig.text(0.82, 0.94, f"Report Date  {report_date}", ha="left", va="center", fontsize=9)
    fig.text(0.96, 0.96, f"Page {page_index} of {total_pages}", ha="right", fontsize=9)

    for offset, (ax, window_start) in enumerate(zip(axes, starts)):
        plot_day_panel(
            ax=ax,
            data=data,
            window_start=window_start,
            day_number=first_day_index + offset,
            show_legend=offset == 0,
            activity_scale=activity_scale,
            lux_log_scale_max=lux_log_scale_max,
        )

    pdf.savefig(fig)
    plt.close(fig)


def generate_sleep_report_pdf(
    input_csv_path: Path,
    output_pdf_path: Path | None = None,
    title: str = DEFAULT_TITLE,
    report_date: str | None = None,
    day_start_hour: int = DEFAULT_DAY_START_HOUR,
    days_per_page: int = DEFAULT_DAYS_PER_PAGE,
    activity_scale: float | None = None,
    lux_log_scale_max: float = DEFAULT_LUX_LOG_SCALE_MAX,
    verbose: bool = False,
) -> Path:
    input_csv_path = Path(input_csv_path)
    output_pdf_path = (
        default_report_output_path(input_csv_path)
        if output_pdf_path is None
        else ensure_pdf_path(Path(output_pdf_path))
    )

    if days_per_page < 1:
        raise ValueError("--days-per-page must be greater than or equal to 1.")

    if verbose:
        print(f"Loading actigraphy CSV: {input_csv_path}")
    data = load_actigraphy_csv(input_csv_path)
    starts = day_window_starts(data=data, day_start_hour=day_start_hour)
    total_pages = 1 + math.ceil(len(starts) / days_per_page)
    report_date = report_date_text(report_date)
    activity_scale = resolve_activity_scale(data, activity_scale)
    lux_log_scale_max = resolve_lux_log_scale(lux_log_scale_max)

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Rendering {len(starts)} daily window(s) to PDF")
        print(f"Activity scale: 0-{activity_scale}")
        print(f"Lux log scale: 0-{lux_log_scale_max}")
    with PdfPages(output_pdf_path) as pdf:
        render_title_page(
            pdf=pdf,
            title=title,
            report_date=report_date,
            input_csv_path=input_csv_path,
            day_count=len(starts),
            total_pages=total_pages,
        )

        for chunk_start in range(0, len(starts), days_per_page):
            chunk = starts[chunk_start : chunk_start + days_per_page]
            page_index = 2 + chunk_start // days_per_page
            render_day_page(
                pdf=pdf,
                data=data,
                starts=chunk,
                page_index=page_index,
                total_pages=total_pages,
                title=title,
                report_date=report_date,
                first_day_index=chunk_start + 1,
                activity_scale=activity_scale,
                lux_log_scale_max=lux_log_scale_max,
            )

    if verbose:
        print(f"Sleep report saved to: {output_pdf_path}")

    return output_pdf_path
