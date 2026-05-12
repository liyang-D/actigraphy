from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def clean_value(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    return value


def split_key_value(line: str, separator: str = ":") -> tuple[str, str | None]:
    if separator not in line:
        return line.strip(), None

    key, value = line.split(separator, 1)
    return key.strip(), clean_value(value)


def parse_key_value_lines(
    lines: list[str],
    separator: str = ":",
) -> dict[str, str | None]:
    fields: dict[str, str | None] = {}

    for line in lines:
        stripped = line.strip()

        if not stripped or separator not in stripped:
            continue

        key, value = split_key_value(stripped, separator=separator)
        fields[key] = value

    return fields


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_first_float(value: str | None) -> float | None:
    if value is None:
        return None

    match = NUMBER_PATTERN.search(value)
    if not match:
        return None

    return float(match.group(0))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_yes_no(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"yes", "true", "1", "on"}:
        return True

    if normalized in {"no", "false", "0", "off"}:
        return False

    raise ValueError("value must be one of yes/no, true/false, 1/0, or on/off.")


def ensure_csv_path(path: Path, label: str = "CSV path") -> Path:
    path = Path(path)

    if path.suffix.lower() != ".csv":
        raise ValueError(f"{label} must end with .csv: {path}")

    return path


def metadata_path_for_csv(csv_path: Path) -> Path:
    csv_path = ensure_csv_path(Path(csv_path))
    return csv_path.with_suffix(".metadata.json")


def parse_timestamp(value: str) -> datetime:
    value = value.strip()

    for fmt in (
        "%Y-%m-%d %H:%M:%S:%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: {value}")


def format_timestamp_millis(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]


def normalize_epoch_label(epoch: str) -> str:
    return epoch.strip().replace(" ", "")
