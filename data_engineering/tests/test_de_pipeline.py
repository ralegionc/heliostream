"""Offline tests for the data-engineering layer (no broker, no network)."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import duckdb

from heliostream.data import synthetic, features as F
from heliostream_de import warehouse as W
from heliostream_de.bus import FileBus


def test_filebus_roundtrip_and_offset(tmp_path):
    bus = FileBus(path=tmp_path / "topic.log", group="t")
    for i in range(5):
        bus.produce({"time": f"2020-01-01T0{i}:00:00", "v": 400 + i})
    first = bus.consume()
    assert len(first) == 5
    assert bus.consume() == []                 # offset committed; nothing new
    bus.produce({"time": "2020-01-01T05:00:00", "v": 999})
    assert len(bus.consume()) == 1             # only the new record


def test_warehouse_upsert_is_idempotent(tmp_path):
    con = W.connect(tmp_path / "wh.duckdb")
    W.init_raw(con)
    recs = [{"time": "2020-01-01T00:00:00", "bt": 5, "bz": -2, "by": 1,
             "v": 400, "n": 5, "dst": -10, "source": "synthetic"}]
    W.upsert_records(con, recs)
    W.upsert_records(con, recs)                # same key again
    n = con.execute(f"SELECT count(*) FROM {W.C.RAW_SCHEMA}.{W.C.RAW_TABLE}").fetchone()[0]
    assert n == 1                              # last-write-wins, no duplicate


def test_feature_parity_sql_vs_pandas():
    """The dbt feature SQL must reproduce heliostream's pandas engineer()."""
    df = synthetic.simulate(300, seed=1)
    r = df.reset_index().rename(columns={"time": "time", "by": "by_gsm"})
    con = duckdb.connect()
    con.register("r", r)
    sql = con.execute("""
        SELECT
            greatest(0.0, -bz)                  AS bs,
            v * greatest(0.0, -bz) * 0.001      AS vbs,
            1.6726e-6 * n * v * v               AS pdyn,
            pow(sin(atan2(by_gsm, bz)/2.0), 4)  AS sin_clock_half4
        FROM r ORDER BY time
    """).df()
    eng = F.engineer(df)
    for col in ["bs", "vbs", "pdyn", "sin_clock_half4"]:
        assert np.allclose(sql[col].values, eng[col].values, rtol=1e-6, atol=1e-9), col
