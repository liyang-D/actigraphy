from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class GeneActiveHeader:
    metadata: dict[str, Any]
    decoder_context: dict[str, Any]
    next_line_index: int


@dataclass
class GeneActivePageHeader:
    sequence_number: int
    page_time: datetime
    temperature: float | None
    battery_voltage: float | None
    device_status: str | None
    measurement_frequency_hz: float


@dataclass
class GeneActivePage:
    header: GeneActivePageHeader
    hex_data: str
