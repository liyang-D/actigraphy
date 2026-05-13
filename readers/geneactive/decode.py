from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterator

from .models import GeneActivePage


SAMPLE_HEX_LENGTH = 12


def signed12(hex_value: str) -> int:
    value = int(hex_value, 16)
    return value - 4096 if value >= 2048 else value


def require_number(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required decoder value: {name}")

    return float(value)


def calibrate_axis(raw_value: int, gain: float, offset: float) -> float:
    return (raw_value * 100 - offset) / gain


def decode_light(
    raw_light_button: int,
    lux_factor: float | None,
    volts_factor: float | None,
) -> int:
    light_output = raw_light_button >> 2

    if lux_factor is None or volts_factor is None or volts_factor == 0:
        lux = float(light_output)
    else:
        factor = lux_factor / volts_factor
        # GENEActiv 1.2 compresses the light sensor's wider range into 10 bits.
        if light_output < 256:
            lux = light_output * factor
        elif light_output < 512:
            lux = (light_output - 128) * 2 * factor
        elif light_output < 768:
            lux = (light_output - 320) * 4 * factor
        elif light_output < 1024:
            lux = (light_output - 656) * 16 * factor
        else:
            lux = 5888 * factor

    return int(lux)


def decode_button(raw_light_button: int) -> int:
    return (raw_light_button >> 1) & 1


def decode_sample(
    sample_hex: str,
    calibration: dict[str, Any],
) -> dict[str, float | int]:
    if len(sample_hex) != SAMPLE_HEX_LENGTH:
        raise ValueError(
            f"Expected {SAMPLE_HEX_LENGTH} hex characters per sample, "
            f"got {len(sample_hex)}."
        )

    x_raw = signed12(sample_hex[0:3])
    y_raw = signed12(sample_hex[3:6])
    z_raw = signed12(sample_hex[6:9])

    light_button_raw = int(sample_hex[9:12], 16)

    x_gain = require_number(calibration.get("x_gain"), "x_gain")
    x_offset = require_number(calibration.get("x_offset"), "x_offset")
    y_gain = require_number(calibration.get("y_gain"), "y_gain")
    y_offset = require_number(calibration.get("y_offset"), "y_offset")
    z_gain = require_number(calibration.get("z_gain"), "z_gain")
    z_offset = require_number(calibration.get("z_offset"), "z_offset")

    lux_factor = calibration.get("lux")
    volts_factor = calibration.get("volts")

    lux_factor = float(lux_factor) if lux_factor is not None else None
    volts_factor = float(volts_factor) if volts_factor is not None else None

    return {
        "Ax": calibrate_axis(x_raw, x_gain, x_offset),
        "Ay": calibrate_axis(y_raw, y_gain, y_offset),
        "Az": calibrate_axis(z_raw, z_gain, z_offset),
        "Lux": decode_light(light_button_raw, lux_factor, volts_factor),
        "Button": decode_button(light_button_raw),
    }


def format_geneactive_time(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]


def decode_page_columns(
    page: GeneActivePage,
    decoder_context: dict[str, Any],
    mode: str = "full",
) -> dict[str, list[Any]]:
    if mode not in {"motion", "full"}:
        raise ValueError("mode must be either 'motion' or 'full'.")

    calibration = decoder_context.get("calibration", {})
    samples_per_page = int(decoder_context.get("samples_per_page", 300))

    sample_rate = page.header.measurement_frequency_hz
    if sample_rate <= 0:
        raise ValueError(f"Invalid page sample rate: {sample_rate}")

    expected_hex_length = samples_per_page * SAMPLE_HEX_LENGTH

    if len(page.hex_data) < expected_hex_length:
        raise ValueError(
            f"Page {page.header.sequence_number} hex data is shorter than expected: "
            f"{len(page.hex_data)} < {expected_hex_length}."
        )

    hex_data = page.hex_data[:expected_hex_length]

    x_gain = require_number(calibration.get("x_gain"), "x_gain")
    x_offset = require_number(calibration.get("x_offset"), "x_offset")
    y_gain = require_number(calibration.get("y_gain"), "y_gain")
    y_offset = require_number(calibration.get("y_offset"), "y_offset")
    z_gain = require_number(calibration.get("z_gain"), "z_gain")
    z_offset = require_number(calibration.get("z_offset"), "z_offset")

    columns: dict[str, list[Any]] = {
        "Time": [],
        "Ax": [],
        "Ay": [],
        "Az": [],
    }

    if mode == "full":
        lux_factor = calibration.get("lux")
        volts_factor = calibration.get("volts")
        lux_factor = float(lux_factor) if lux_factor is not None else None
        volts_factor = float(volts_factor) if volts_factor is not None else None

        columns["Lux"] = []
        columns["Button"] = []
        columns["Temperature"] = []

    dt_seconds = 1.0 / sample_rate

    for sample_index in range(samples_per_page):
        start = sample_index * SAMPLE_HEX_LENGTH
        sample_hex = hex_data[start : start + SAMPLE_HEX_LENGTH]

        sample_time = page.header.page_time + timedelta(
            seconds=sample_index * dt_seconds
        )
        columns["Time"].append(format_geneactive_time(sample_time))
        columns["Ax"].append(
            calibrate_axis(signed12(sample_hex[0:3]), x_gain, x_offset)
        )
        columns["Ay"].append(
            calibrate_axis(signed12(sample_hex[3:6]), y_gain, y_offset)
        )
        columns["Az"].append(
            calibrate_axis(signed12(sample_hex[6:9]), z_gain, z_offset)
        )

        if mode == "full":
            light_button_raw = int(sample_hex[9:12], 16)
            columns["Lux"].append(
                decode_light(light_button_raw, lux_factor, volts_factor)
            )
            columns["Button"].append(decode_button(light_button_raw))
            columns["Temperature"].append(page.header.temperature)

    return columns


def decode_page(
    page: GeneActivePage,
    decoder_context: dict[str, Any],
    mode: str = "full",
) -> Iterator[dict[str, Any]]:
    columns = decode_page_columns(
        page=page,
        decoder_context=decoder_context,
        mode=mode,
    )
    output_columns = ["Time", "Ax", "Ay", "Az"]
    if mode == "full":
        output_columns.extend(["Lux", "Button", "Temperature"])

    row_count = len(columns["Time"])
    for row_index in range(row_count):
        yield {
            column: columns[column][row_index]
            for column in output_columns
        }


def decode_pages(
    pages,
    decoder_context: dict[str, Any],
    mode: str = "full",
) -> Iterator[dict[str, Any]]:
    for page in pages:
        yield from decode_page(
            page=page,
            decoder_context=decoder_context,
            mode=mode,
        )
