from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar


READER_OUTPUT_COLUMNS: dict[str, list[str]] = {
    "motion": ["Time", "Ax", "Ay", "Az"],
    "full": ["Time", "Ax", "Ay", "Az", "Lux", "Button", "Temperature"],
}


def get_reader_output_columns(mode: str) -> list[str]:
    try:
        return list(READER_OUTPUT_COLUMNS[mode])
    except KeyError as exc:
        raise ValueError("mode must be either 'motion' or 'full'.") from exc


def default_raw_output_csv_path(
    input_path: Path,
    output_dir: Path | None = None,
) -> Path:
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent

    return output_dir / f"{input_path.stem}_raw.csv"


def default_metadata_path(output_csv_path: Path) -> Path:
    return Path(output_csv_path).with_suffix(".metadata.json")


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


class BaseDeviceReader(ABC):
    name: ClassVar[str]
    supported_extensions: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def read(
        self,
        input_path: Path,
        output_csv_path: Path | None = None,
        output_metadata_path: Path | None = None,
        output_dir: Path | None = None,
        mode: str = "full",
        max_pages: int | None = None,
        verbose: bool = False,
    ) -> tuple[Path, Path]:
        """Read a device file and write Step 1A CSV and metadata outputs."""

    def load_samples(
        self,
        input_path: Path,
        mode: str = "full",
        max_pages: int | None = None,
        verbose: bool = False,
    ) -> Any:
        """Load a device file as standard sample-level data without writing files."""
        raise NotImplementedError(f"{self.name} does not support in-memory loading.")
