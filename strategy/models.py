"""
strategy/models.py
------------------
Standardized models for the strategy logic.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Direction(str, Enum):
    LONG  = "long"  # buy
    SHORT = "short"  # sell
    FLAT  = "flat"  # neither bought nor sold
    NAN = "nan"  # not calculatable / error


@dataclass
class Signal:
    direction: Direction
    strength:  float  # indicator between 0 and 1
    date: date | None
    strategy:  str
    metadata:  dict = field(default_factory=dict)
