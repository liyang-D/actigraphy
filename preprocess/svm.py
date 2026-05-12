from __future__ import annotations

import numpy as np
import pandas as pd


SVM_SAMPLE_COLUMN = "_svm_sample"


def add_geneactive_svm(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    vector_magnitude = np.sqrt(data["Ax"] ** 2 + data["Ay"] ** 2 + data["Az"] ** 2)
    data[SVM_SAMPLE_COLUMN] = np.abs(vector_magnitude - 1.0)

    return data
