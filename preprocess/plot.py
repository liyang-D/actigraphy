# preprocess/plot.py

import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from models import EpochSVMData


def plot_svm_timeseries(svm_data: EpochSVMData, out_file: str) -> None:
    svm_data.validate()

    df = svm_data.data.copy()

    if "Time" not in df.columns:
        df = df.reset_index()

    if "Time" not in df.columns:
        raise ValueError("Plotting requires time information in the index or a 'Time' column.")

    r, _ = df.shape
    if r <= 0:
        raise ValueError("Cannot plot an empty dataframe.")

    order_of_magnitude_rows = math.floor(math.log10(r))

    if order_of_magnitude_rows <= 2:
        locx = 10
    else:
        locx = 100

    fig, ax = plt.subplots()
    ax.plot(df["Time"], df["SVM"], "b-")
    ax.xaxis.set_major_locator(MultipleLocator(locx))
    plt.xticks(rotation=90)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)