"""Heliostream warehouse dashboard.

A read-only view over the DuckDB feature mart (`main.features_hourly`): storm
history, severity mix, feed coverage, recent solar wind, and the data-quality
gate. Complements the model's nowcast page; this one visualizes the data the
pipeline produces.

Run:
    cd data_engineering/dashboard
    streamlit run warehouse_app.py
    # point at a specific warehouse:  HELIO_DUCKDB=/path/to.duckdb streamlit run warehouse_app.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Load the sibling queries.py by explicit path. Importing it by bare name would
# let any same-named module elsewhere on sys.path (there is a `queries` package
# on PyPI) shadow it, which silently yields a module missing our functions.
_qpath = Path(__file__).resolve().parent / "queries.py"
_spec = importlib.util.spec_from_file_location("helio_queries", _qpath)
Q = importlib.util.module_from_spec(_spec)
sys.modules["helio_queries"] = Q
_spec.loader.exec_module(Q)

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Heliostream warehouse", page_icon="*",
                   layout="wide", initial_sidebar_state="collapsed")

INDIGO, VIOLET, TEAL, AMBER, ORANGE, RED = (
    "#7c5cff", "#a78bfa", "#3ee6c4", "#c9a227", "#ff8c42", "#ff4d6d")
GRID, INK, DIM = "#211a3d", "#e9e8f2", "#8b88a8"

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .block-container { padding-top: 2rem; max-width: 1250px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  [data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }
  .helio-title { font-size: 26px; font-weight: 700; }
  .helio-sub { color: #5a5878; font-size: 12px; letter-spacing: .16em;
               text-transform: uppercase; }
  .badge { display:inline-block; padding:3px 11px; border-radius:999px;
           font-size:12px; border:1px solid #2c2450; }
</style>
""", unsafe_allow_html=True)


def dark(chart, h=260):
    return (chart.properties(height=h, background="rgba(0,0,0,0)")
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelColor=DIM, titleColor=DIM, labelFontSize=11)
            .configure_legend(labelColor=DIM, titleColor=DIM))


# ------------------------------------------------------------------ data load
@st.cache_data(ttl=30)
def load(db_path: str | None):
    con = Q.connect(db_path)
    if not Q.has_features(con):
        return None
    return {
        "kpis": Q.kpis(con),
        "daily": Q.dst_daily(con),
        "severity": Q.severity_distribution(con),
        "sources": Q.source_coverage(con),
        "recent": Q.recent_solar_wind(con, 168),
        "monthly": Q.monthly_storm_counts(con),
        "quality": Q.quality_checks(con),
        "gaps": Q.gap_summary(con),
        "gap_list": Q.largest_gaps(con, 8),
    }


db_path = st.sidebar.text_input("Warehouse (DuckDB) path", value=str(Q.default_db_path()))
if st.sidebar.button("Refresh"):
    st.cache_data.clear()
st.sidebar.caption("Read-only view of main.features_hourly. Auto-caches for 30s.")

try:
    data = load(db_path or None)
except Exception as e:
    st.error(f"**{type(e).__name__}:** {e}")
    st.caption(f"queries module loaded from: `{_qpath}`")
    with st.expander("Details"):
        st.exception(e)
    st.stop()

if data is None:
    st.markdown('<div class="helio-title">Heliostream warehouse</div>',
                unsafe_allow_html=True)
    st.warning("No feature mart found yet. Build it first:\n\n"
               "`python -m heliostream_de run-batch --source synthetic --hours 26280`")
    st.stop()

k = data["kpis"]
latest = pd.Timestamp(k["latest"])
now = pd.Timestamp.now(tz="UTC").tz_convert(None)
fresh_h = (now - latest).total_seconds() / 3600
fresh_color = TEAL if fresh_h <= 3 else (AMBER if fresh_h <= 12 else RED)

# ---------------------------------------------------------------------- header
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('<div class="helio-title">Heliostream warehouse</div>'
                '<div class="helio-sub">feature mart &middot; main.features_hourly</div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(
        f'<div style="text-align:right"><span class="badge" '
        f'style="color:{fresh_color};border-color:{fresh_color}">latest '
        f'{latest:%Y-%m-%d %H:%M} &middot; {fresh_h:,.0f} h ago</span></div>',
        unsafe_allow_html=True)

st.write("")
m = st.columns(5)
m[0].metric("Hours", f"{k['rows']:,}")
m[1].metric("Span (days)", f"{k['span_days']:,}")
m[2].metric("Storm hours", f"{k['storm_hours']:,}",
            help="Dst < -50 nT")
m[3].metric("Intense hours", f"{k['intense_hours']:,}", help="Dst < -100 nT")
m[4].metric("Min Dst", f"{k['min_dst']:.0f} nT" if k['min_dst'] is not None else "n/a")

# ------------------------------------------------------------------ Dst timeline
st.subheader("Dst storm history")
daily = data["daily"]
base = alt.Chart(daily).encode(x=alt.X("day:T", title=None))
band = base.mark_area(color=INDIGO, opacity=0.18).encode(
    y=alt.Y("dst_min:Q", title="Dst (nT)"))
line = base.mark_line(color=VIOLET, strokeWidth=1).encode(y="dst_mean:Q")
thresh = alt.Chart(pd.DataFrame({"y": [-50]})).mark_rule(
    color=ORANGE, strokeDash=[4, 4]).encode(y="y:Q")
st.altair_chart(dark(band + line + thresh, 300), width='stretch')
st.caption("Shaded = daily storm depth (most negative Dst). Line = daily mean. "
           "Dashed = storm threshold (-50 nT).")

# ------------------------------------------------ severity + source coverage
left, right = st.columns(2)
with left:
    st.subheader("Storm severity mix")
    sev = data["severity"]
    palette = [TEAL, "#8ab4ff", AMBER, ORANGE, RED]
    chart = alt.Chart(sev).mark_bar().encode(
        x=alt.X("hours:Q", title="hours"),
        y=alt.Y("severity:N", sort=list(sev["severity"]), title=None),
        color=alt.Color("severity:N",
                        scale=alt.Scale(domain=list(sev["severity"]), range=palette),
                        legend=None),
        tooltip=["severity", "hours"],
    )
    st.altair_chart(dark(chart, 220), width='stretch')

with right:
    st.subheader("Feed coverage by source")
    src = data["sources"]
    donut = alt.Chart(src).mark_arc(innerRadius=55).encode(
        theta="hours:Q",
        color=alt.Color("source:N",
                        scale=alt.Scale(range=[INDIGO, TEAL, AMBER]), title=None),
        tooltip=["source", "hours"],
    )
    st.altair_chart(dark(donut, 220), width='stretch')

# ------------------------------------------------------ monthly storm frequency
st.subheader("Storm hours per month")
mon = data["monthly"]
bars = alt.Chart(mon).mark_bar(color=INDIGO).encode(
    x=alt.X("month:T", title=None),
    y=alt.Y("storm_hours:Q", title="storm hours"),
    tooltip=["month:T", "storm_hours:Q"])
st.altair_chart(dark(bars, 200), width='stretch')

# ------------------------------------------------------ recent solar wind + QA
lw, rw = st.columns([2, 1])
with lw:
    st.subheader("Recent solar wind (last 7 days in mart)")
    rec = data["recent"]
    bz = alt.Chart(rec).mark_line(color=VIOLET).encode(
        x=alt.X("time:T", title=None), y=alt.Y("bz:Q", title="Bz (nT)"))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=DIM, opacity=0.5).encode(y="y:Q")
    st.altair_chart(dark(bz + zero, 150), width='stretch')
    v = alt.Chart(rec).mark_line(color=TEAL).encode(
        x=alt.X("time:T", title=None), y=alt.Y("v:Q", title="V (km/s)"))
    st.altair_chart(dark(v, 150), width='stretch')

with rw:
    st.subheader("Data-quality gate")
    qa = data["quality"]
    for _, r in qa.iterrows():
        mark = "✅" if r["pass"] else "❌"
        st.markdown(
            f"{mark}&nbsp; **{r['check']}**  \n"
            f"<span style='color:{DIM};font-size:12px'>{r['detail']}</span>",
            unsafe_allow_html=True)
    if bool(qa["pass"].all()):
        st.success("All checks passing")
    else:
        st.error("Quality gate failing")

# ------------------------------------------------------------ feed continuity
g = data["gaps"]
st.subheader("Feed continuity")
gc = st.columns(4)
gc[0].metric("Coverage", f"{1 - g['missing_fraction']:.2%}")
gc[1].metric("Gaps", f"{g['n_gaps']:,}", help="Runs of missing hours (> 1 h)")
gc[2].metric("Missing hours", f"{g['missing_hours']:,}")
gc[3].metric("Largest outage", f"{g['max_gap']:,} h")
if not data["gap_list"].empty:
    with st.expander("Largest outages"):
        st.dataframe(data["gap_list"], width='stretch', hide_index=True)
st.caption("Spacecraft telemetry always has some dropout. Gaps are reported as "
           "information; the gate fails only if coverage drops below 98% or a "
           "single outage exceeds 72 h.")

st.caption("Heliostream data-engineering layer &middot; DuckDB + dbt &middot; "
           "read-only dashboard over the tested feature mart.")
