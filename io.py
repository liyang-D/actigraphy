import pandas as pd

from models import EpochSVMData, RawTriAxialData


def load_raw_csv(path: str, sample_rate: float | None = None) -> RawTriAxialData:
    data = pd.read_csv(path, names=["Time", "Ax", "Ay", "Az"])
    raw_data = RawTriAxialData(data=data, sample_rate=sample_rate)
    raw_data.validate()
    return raw_data


def save_epoch_svm_csv(data: EpochSVMData, path: str) -> None:
    data.validate()
    data.data.to_csv(path, index=True)