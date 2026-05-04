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
    FLAT    = "flat"     # not buy nor sell
    INVALID = "invalid"  # could not be computed (missing data / error)


@dataclass
class Signal:
    direction: Direction
    strength:  float | None  # 0.0 - 1.0 for LONG/SHORT; None for FLAT/INVALID
    date:      date | None
    strategy:  str
    metadata:  dict = field(default_factory=dict)

    def __str__(self) -> str:
        strength_str = f"{self.strength:.4f}" if self.strength is not None else "—"
        date_str = self.date.strftime("%d.%m.%Y") if self.date else "—"
        reasons = self.metadata.get("reasons") or self.metadata.get("error")
        reason_str = f"\n  reasons: {reasons}" if reasons else ""
        
        return (
            f"Signal("
            f"direction={self.direction.value:<7} "
            f"strength={strength_str}  "
            f"date={date_str}  "
            f"strategy={self.strategy}"
            f"{reason_str})"
        )
