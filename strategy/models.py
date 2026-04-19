"""
strategy/models.py
------------------
Standardized models for the strategy logic.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Direction(str, Enum):
    LONG    = "long"     # buy signal
    SHORT   = "short"    # sell signal
    FLAT    = "flat"     # no position recommended
    INVALID = "invalid"  # could not be computed (missing data / error)


@dataclass
class Signal:
    direction: Direction
    strength:  float | None  # 0.0 - 1.0 for LONG/SHORT; None for FLAT/INVALID
    date:      date | None
    strategy:  str
    metadata:  dict = field(default_factory=dict)
