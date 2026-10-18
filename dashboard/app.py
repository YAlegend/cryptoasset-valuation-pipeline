"""Cryptoasset valuation dashboard (spec 12 deliverable 3, Phase 5). Run: `make dashboard`.

Presentation only: every number comes from cryptoval.dashboard.queries over the dbt marts.
Each read opens a read-only DuckDB connection and closes it at once; holding one open would
take the file lock that `make all`'s dbt writes need.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from cryptoval.dashboard import queries as q

DISCLAIMER = (
    "**These are valuation estimates under stated assumptions, not investment advice. These "
    "assets are volatile and these methods disagree by design.**"
)
CLIP = 2.0  # heatmap colour clip, in orders of magnitude


def _db_path() -> str:
    return str(pathlib.Path(os.environ.get("CRYPTOVAL_DATA_ROOT", "data")) / "cryptoval.duckdb")


@st.cache_data(ttl=600, show_spinner=False)
def _read(db: str, name: str, *args):
    con = duckdb.connect(db, read_only=True)
    try:
        return getattr(q, name)(con, *args)
    finally:
        con.close()


def _symbols(db: str) -> list[str]:
    con = duckdb.connect(db, read_only=True)
    try:
        return [
            r[0]
            for r in con.execute(
                "select distinct symbol from fct_valuation_estimate order by symbol"
            ).fetchall()
        ]
    finally:
        con.close()


def _banner(db: str) -> None:
    fresh = _read(db, "freshness", dt.date.today())
    coin = fresh["coin_age_through"] or "not extracted"
    text = (
        f"Panel through **{fresh['panel_through']}** (vintage {fresh['panel_vintage']}); "
        f"BTC coin-age data through **{coin}**."
    )
    if fresh["stale"]:
        st.warning(f"{text} The panel is {fresh['panel_lag_days']} days old: run `make all`.")
    else:
        st.caption(text)


def _football_field(db: str, symbol: str) -> None:
    field = _read(db, "football_field", symbol)
    price = _read(db, "latest_price", symbol)
    if field.empty:
        st.info(f"No method produces a fair value for {symbol}.")
    else:
        field = field.assign(
            basis=field["is_circular"].map(
                {True: "circular (reads market cap)", False: "independent"}
            ),
            label=field["method_id"] + " (" + field["metric_date"].astype(str) + ")",
        )
        x = alt.X("low:Q", title="fair value, USD (log scale)", scale=alt.Scale(type="log"))
        bars = (
            alt.Chart(field)
            .mark_bar(size=12)
            .encode(
                x=x,
                x2="high:Q",
                y=alt.Y("label:N", sort=None, title=None),
                color=alt.Color("basis:N", title=None),
                tooltip=["method_id", "family", "metric_date", "low", "mid", "high", "n_cells"],
            )
        )
        ticks = (
            alt.Chart(field)
            .mark_tick(thickness=2, size=16, color="black")
            .encode(x="mid:Q", y=alt.Y("label:N", sort=None))
        )
        layers = [bars, ticks]
        if price is not None:
            rule = pd.DataFrame({"price": [price[1]], "label": [f"price {price[0]}"]})
            layers.append(
                alt.Chart(rule)
                .mark_rule(color="red", strokeDash=[4, 2])
                .encode(x="price:Q", tooltip=["label", "price"])
            )
        st.altair_chart(alt.layer(*layers).properties(height=26 * len(field) + 40), width="stretch")
        if price is not None:
            st.caption(
                f"Red line: {symbol} price ${price[1]:,.2f} on {price[0]}. Ticks: grid median."
            )
    refused = _read(db, "refusals", symbol)
    if not refused.empty:
        with st.expander(f"{len(refused)} method(s) never apply to {symbol}"):
            st.dataframe(refused, hide_index=True, width="stretch")


def _indicators(db: str) -> None:
    name = st.selectbox("Indicator (BTC)", q.INDICATORS)
    series = _read(db, "indicator_series", name)
    if series.empty:
        st.info(f"No {name} history yet.")
        return
    chart = (
        alt.Chart(series)
        .mark_line()
        .encode(x=alt.X("metric_date:T", title=None), y=alt.Y("value:Q", title=name))
    )
    layers = [chart]
    for level in q.THRESHOLDS.get(name, []):
        layers.append(
            alt.Chart(pd.DataFrame({"y": [level]}))
            .mark_rule(color="red", strokeDash=[4, 2])
            .encode(y="y:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")
    note = q.THRESHOLDS.get(name)
    st.caption(
        f"Latest {series['value'].iloc[-1]:.4g} on {series['metric_date'].iloc[-1]}."
        + (f" Red: the dissertation's stated thresholds {note}." if note else "")
    )


def _agreement(db: str) -> None:
    grid = _read(db, "agreement")
    if grid.empty:
        st.info("No fair values to compare.")
        return
    grid = grid.assign(shown=grid["log_ratio"].clip(-CLIP, CLIP))
    chart = (
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("method_id:N", title=None),
            y=alt.Y("symbol:N", title=None),
            color=alt.Color(
                "shown:Q",
                title="log10(fair / price)",
                scale=alt.Scale(scheme="redblue", domain=[-CLIP, CLIP], reverse=True),
            ),
            tooltip=[
                "symbol",
                "method_id",
                "metric_date",
                alt.Tooltip("log_ratio:Q", format=".2f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Blue: method says undervalued; red: overvalued. Colour clipped at ±{CLIP:g} orders of "
        f"magnitude (raw range {grid['log_ratio'].min():.1f} to {grid['log_ratio'].max():.1f}). "
        "Each cell is the method's grid median on its latest applicable day, against that "
        "day's price."
    )


def _dcf(db: str) -> None:
    rows = _read(db, "dcf_inputs")
    if rows.empty:
        st.info("No applicable DCF row to start from.")
        return
    keys = rows["symbol"] + " / " + rows["method_id"]
    choice = st.selectbox("Asset and DCF", keys.tolist())
    row = rows[keys == choice].iloc[0]
    growth = list(row["growth"])
    cols = st.columns(2)
    discount = cols[0].slider("Discount rate", 0.01, 0.40, float(row["discount_rate"]), 0.01)
    terminal = cols[1].slider(
        "Terminal multiple (0 = Gordon growth)",
        0.0,
        40.0,
        float(row["terminal_multiple"] or 0.0),
        1.0,
    )
    g = st.columns(3)
    growth = [
        g[i].slider(f"Stage {i + 1} growth", -0.20, 0.60, float(growth[i]), 0.01) for i in range(3)
    ]
    price = _read(db, "latest_price", row["symbol"])
    try:
        value, share = q.dcf_value(
            float(row["annual_cash_flow_usd"]),
            float(row["supply"]),
            growth,
            discount,
            list(row["stage_years"]),
            terminal or None,
        )
    except ValueError as exc:
        st.error(f"No fair value at these settings: {exc}")
        return
    m = st.columns(3)
    m[0].metric("Fair value per token", f"${value:,.2f}")
    m[1].metric("Latest price", f"${price[1]:,.2f}" if price else "n/a")
    m[2].metric("Terminal share of value", f"{share:.0%}")
    st.caption(
        f"Annual cash flow ${row['annual_cash_flow_usd']:,.0f} and supply {row['supply']:,.0f} "
        f"from the {row['method_id']} row of {row['metric_date']}. "
        f"Stage years {row['stage_years']}."
        + (
            " Terminal value exceeds 80% of the total: the answer is mostly the terminal "
            "assumption."
            if share > 0.8
            else ""
        )
    )


def main() -> None:
    st.set_page_config(page_title="Cryptoasset valuation", layout="wide")
    st.title("Cryptoasset valuation")
    st.markdown(DISCLAIMER)
    db = _db_path()
    if not pathlib.Path(db).exists():
        st.error(f"No database at {db}. Build it with `make all`, then reload.")
        return
    _banner(db)
    tabs = st.tabs(["Football field", "Cycle indicators", "Method agreement", "DCF sliders"])
    with tabs[0]:
        symbols = _symbols(db)
        symbol = st.selectbox(
            "Asset", symbols, index=symbols.index("BTC") if "BTC" in symbols else 0
        )
        _football_field(db, symbol)
    with tabs[1]:
        _indicators(db)
    with tabs[2]:
        _agreement(db)
    with tabs[3]:
        _dcf(db)


main()
