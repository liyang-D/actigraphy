def clean_value(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    return value


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        return float(value.strip())
    except ValueError:
        return None


