# preprocess/epoch.py

import pandas as pd

from models import EpochSVMData


def epoch_svm(data_with_svm: pd.DataFrame, binsize: str) -> EpochSVMData:
    sub_sampled = data_with_svm.resample(binsize).mean()
    return EpochSVMData(data=sub_sampled, binsize=binsize)