from .base import (
    BaseDeviceReader,
    default_metadata_path,
    default_raw_output_csv_path,
    get_reader_output_columns,
)
from .geneactive import GeneActiveReader


READERS: dict[str, BaseDeviceReader] = {
    GeneActiveReader.name: GeneActiveReader(),
}


def available_readers() -> tuple[str, ...]:
    return tuple(sorted(READERS))


def get_reader(name: str) -> BaseDeviceReader:
    try:
        return READERS[name]
    except KeyError as exc:
        available = ", ".join(available_readers())
        raise ValueError(
            f"Unknown reader {name!r}. Available readers: {available}"
        ) from exc


__all__ = [
    "BaseDeviceReader",
    "GeneActiveReader",
    "available_readers",
    "default_metadata_path",
    "default_raw_output_csv_path",
    "get_reader",
    "get_reader_output_columns",
]
