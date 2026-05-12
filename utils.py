from __future__ import annotations

import re


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

