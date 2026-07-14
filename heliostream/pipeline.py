"""Dataset acquisition + caching. Dispatches by source and caches to parquet."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .data import synthetic


def load_dataset(source="synthetic", refresh=False, **kw) -> pd.DataFrame:
    """Return a tidy hourly frame [bt,bz,by,v,n,dst]. Cached under artifacts/data."""
    cache = C.DATA_DIR / f"{source}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    if source == "synthetic":
        years = kw.get("years", 6)
        df = synthetic.simulate(hours=int(24 * 365 * years),
                                seed=kw.get("seed", C.SEED))
    elif source == "omni":
        from .data import omni  # imported lazily (needs internet)
        df = omni.fetch(start=kw.get("start", "2016-01-01"),
                        stop=kw.get("stop", "2024-01-01"))
    elif source == "warehouse":
        # Read the dbt-built feature mart from the DuckDB warehouse instead of an
        # in-memory frame. Requires the data_engineering pipeline to have run.
        import duckdb
        db = kw.get("duckdb_path")
        if db is None:
            raise ValueError("source='warehouse' needs duckdb_path=...")
        relation = kw.get("relation", "main.features_hourly")
        con = duckdb.connect(str(db), read_only=True)
        df = con.execute(
            f"SELECT time, bt, bz, by_gsm AS by, v, n, dst FROM {relation} "
            f"WHERE dst IS NOT NULL ORDER BY time"
        ).df()
        con.close()
        df["time"] = __import__("pandas").to_datetime(df["time"])
        df = df.set_index("time")
        return df  # already tidy; skip parquet cache (warehouse is the source of truth)
    else:
        raise ValueError(f"unknown source '{source}'")

    df.to_parquet(cache)
    return df
