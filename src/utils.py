"""Small helpers shared across modules."""

import json
from pathlib import Path
from typing import Any

from src import config

MILLION = 1_000_000
THOUSAND = 1_000


def format_currency(value: float) -> str:
    """Format a monetary value compactly, for example `₹1.25M` or `₹60K`.

    Args:
        value: Amount in currency units.

    Returns:
        A short human-readable string using the configured currency symbol.
    """
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= MILLION:
        return f"{sign}{config.CURRENCY_SYMBOL}{magnitude / MILLION:.2f}M"
    if magnitude >= THOUSAND:
        return f"{sign}{config.CURRENCY_SYMBOL}{magnitude / THOUSAND:.0f}K"
    return f"{sign}{config.CURRENCY_SYMBOL}{magnitude:.0f}"


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a dictionary to disk as indented JSON, creating folders as needed.

    Args:
        payload: JSON-serialisable data.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
