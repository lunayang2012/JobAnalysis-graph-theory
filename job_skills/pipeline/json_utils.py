from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_default(o: Any) -> Any:
    """Make common pandas/numpy/path/datetime objects JSON-serializable."""
    # Pandas missing
    if o is pd.NaT:
        return None
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass

    # Datetimes
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()

    # NumPy scalars
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)

    # Paths
    if isinstance(o, Path):
        return str(o)

    # Enums
    if isinstance(o, Enum):
        return o.value

    return str(o)
