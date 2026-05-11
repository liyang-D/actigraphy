from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import GeneActiveHeader
from utils import clean_value, parse_float, parse_int


def split_key_value(line: str) -> tuple[str, str | None]:
    if ":" not in line:
        return line.strip(), None

    key, value = line.split(":", 1)
    return key.strip(), clean_value(value)


def parse_measurement_frequency(value: str | None) -> float | None:
    if not value:
        return None

    match = re.search(r"([-+]?\d+(?:\.\d+)?)", value)
    if not match:
        return None

    return float(match.group(1))


def read_header_lines(path: Path, n_lines: int = 59) -> list[str]:
    lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(n_lines):
            line = f.readline()

            if line == "":
                break

            lines.append(line.rstrip("\n\r"))

    return lines


def parse_sensor_info(header_lines: list[str]) -> dict[str, dict[str, str | None]]:
    sensors: dict[str, dict[str, str | None]] = {
        "accelerometer": {
            "range": None,
            "resolution": None,
            "units": None,
        },
        "light_meter": {
            "range": None,
            "resolution": None,
            "units": None,
        },
        "temperature_sensor": {
            "range": None,
            "resolution": None,
            "units": None,
        },
    }

    field_map = {
        "Accelerometer Range": ("accelerometer", "range"),
        "Accelerometer Resolution": ("accelerometer", "resolution"),
        "Accelerometer Units": ("accelerometer", "units"),
        "Light Meter Range": ("light_meter", "range"),
        "Light Meter Resolution": ("light_meter", "resolution"),
        "Light Meter Units": ("light_meter", "units"),
        "Temperature Sensor Range": ("temperature_sensor", "range"),
        "Temperature Sensor Resolution": ("temperature_sensor", "resolution"),
        "Temperature Sensor Units": ("temperature_sensor", "units"),
    }

    for line in header_lines:
        stripped = line.strip()

        if not stripped or ":" not in stripped:
            continue

        key, value = split_key_value(stripped)

        if key not in field_map:
            continue

        sensor_name, field_name = field_map[key]
        sensors[sensor_name][field_name] = value

    return sensors


def parse_geneactive_main_header(path: Path) -> GeneActiveHeader:
    header_lines = read_header_lines(path, n_lines=59)

    raw_fields: dict[str, str | None] = {}

    for line in header_lines:
        stripped = line.strip()

        if not stripped:
            continue

        if ":" not in stripped:
            continue

        key, value = split_key_value(stripped)
        raw_fields[key] = value

    def get(key: str) -> str | None:
        return raw_fields.get(key)

    measurement_frequency_hz = parse_measurement_frequency(
        get("Measurement Frequency")
    )

    sensor_info = parse_sensor_info(header_lines)

    metadata = {
        "source": {
            "input_file": str(path),
            "device_type": get("Device Type"),
            "device_model": get("Device Model"),
            "device_unique_serial_code": get("Device Unique Serial Code"),
            "device_firmware_version": get("Device Firmware Version"),
            "calibration_date": get("Calibration Date"),
            "application_name_and_version": get("Application name & version"),
        },
        "recording": {
            "measurement_frequency_hz": measurement_frequency_hz,
            "measurement_period": get("Measurement Period"),
            "start_time": get("Start Time"),
            "last_measurement": get("Last measurement"),
            "time_zone": get("Time Zone"),
            "device_location_code": get("Device Location Code"),
        },
        "trial": {
            "study_centre": get("Study Centre"),
            "study_code": get("Study Code"),
            "investigator_id": get("Investigator ID"),
            "exercise_type": get("Exercise Type"),
            "config_operator_id": get("Config Operator ID"),
            "config_time": get("Config Time"),
            "config_notes": get("Config Notes"),
            "extract_operator_id": get("Extract Operator ID"),
            "extract_time": get("Extract Time"),
            "extract_notes": get("Extract Notes"),
        },
        "subject": {
            "subject_code": get("Subject Code"),
            "date_of_birth": get("Date of Birth"),
            "sex": get("Sex"),
            "height": get("Height"),
            "weight": get("Weight"),
            "handedness_code": get("Handedness Code"),
            "subject_notes": get("Subject Notes"),
        },
        "sensors": sensor_info,
    }

    decoder_context = {
        "measurement_frequency_hz": measurement_frequency_hz,
        "samples_per_page": 300,
        "start_time": get("Start Time"),
        "time_zone": get("Time Zone"),
        "number_of_pages": parse_int(get("Number of Pages")),
        "calibration": {
            "x_gain": parse_float(get("x gain")),
            "x_offset": parse_float(get("x offset")),
            "y_gain": parse_float(get("y gain")),
            "y_offset": parse_float(get("y offset")),
            "z_gain": parse_float(get("z gain")),
            "z_offset": parse_float(get("z offset")),
            "volts": parse_float(get("Volts")),
            "lux": parse_float(get("Lux")),
        },
    }

    return GeneActiveHeader(
        metadata=metadata,
        decoder_context=decoder_context,
        next_line_index=len(header_lines),
    )


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    test_bin_path = Path("data_test/efthyvoulos__111238_2026-03-06 14-16-51.bin")
    result = parse_geneactive_main_header(test_bin_path)

    output_metadata_path = test_bin_path.with_suffix(".metadata.json")

    save_metadata(result.metadata, output_metadata_path)

    print("Metadata saved to:")
    print(output_metadata_path)

    print("\nDecoder context:")
    print(json.dumps(result.decoder_context, indent=2, ensure_ascii=False))

    print("\nNext line index:")
    print(result.next_line_index)


if __name__ == "__main__":
    main()