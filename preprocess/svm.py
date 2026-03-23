# preprocess/svm.py

import numpy as np
import pandas as pd

from models import RawTriAxialData


def compute_svm(raw_data: RawTriAxialData) -> pd.DataFrame:
    raw_data.validate()

    data = raw_data.data.copy()
    data["Time"] = pd.to_datetime(data["Time"])
    data = data.set_index("Time")

    data["SVM"] = np.maximum(
        0,
        np.sqrt(data["Ax"] ** 2 + data["Ay"] ** 2 + data["Az"] ** 2) - 1,
    )
    return data