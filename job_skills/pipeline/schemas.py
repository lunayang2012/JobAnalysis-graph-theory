from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera.pandas import Column, DataFrameSchema
except ImportError:  # pandera optional, but recommended
    pa = None
    DataFrameSchema = None  # type: ignore[misc]


def get_main_schema() -> Optional["DataFrameSchema"]:
    if pa is None:
        return None

    # Relaxed schema: allows extra columns, enforces types on common ones.
    return DataFrameSchema(
        {
            "title": Column(str, nullable=True, coerce=True),
            "description": Column(str, nullable=True, coerce=True),
            #"cleaned_description": Column(str, nullable=True, coerce=True),
            "company": Column(str, nullable=True, coerce=True),
            "location": Column(str, nullable=True, coerce=True),
            "domain": Column(str, nullable=True, coerce=True),
            "min_amount": Column(float, nullable=True, coerce=True),
            "max_amount": Column(float, nullable=True, coerce=True),
            "currency": Column(str, nullable=True, coerce=True),
            "site": Column(str, nullable=True, coerce=True),
            "job_url": Column(str, nullable=True, coerce=True),
            "date_posted": Column(str, nullable=True, coerce=True),
        },
        coerce=True,
        strict=False,
    )


def validate_main(df: pd.DataFrame) -> pd.DataFrame:
    schema = get_main_schema()
    if schema is None:
        return df
    return schema.validate(df, lazy=True)
