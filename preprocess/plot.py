from __future__ import annotations

import matplotlib.pyplot as plt

from models import EpochSummaryData


def plot_svm_timeseries(summary_data: EpochSummaryData, output_path: str) -> None:
    summary_data.validate()

    data = summary_data.data.copy()
    if "Time" not in data.columns or "SVM_sum" not in data.columns:
        raise ValueError("Plotting requires 'Time' and 'SVM_sum' columns.")

    fig, ax = plt.subplots()
    ax.plot(data["Time"], data["SVM_sum"])
    ax.set_xlabel("Time")
    ax.set_ylabel("SVM_sum")
    plt.xticks(rotation=90)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
