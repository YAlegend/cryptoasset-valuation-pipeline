"""Cryptoasset valuation dashboard (spec 12 deliverable 3, Phase 5). Run: `make dashboard`.

Presentation only: every number comes from cryptoval.dashboard.queries over the dbt marts.
Each read opens a read-only DuckDB connection and closes it at once; holding one open would
take the file lock that `make all`'s dbt writes need.

Two views: a screener of every asset, and one asset's profile at ?asset=SYMBOL.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import tempfile

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from cryptoval.dashboard import queries as q
from cryptoval.dashboard import snapshot

DISCLAIMER = (
    "These are valuation estimates under stated assumptions, not investment advice. These "
    "assets are volatile and these methods disagree by design."
)
CLIP = 2.0  # heatmap colour clip, in orders of magnitude
PRICE = "#f59e0b"  # today's price on the models chart, visible on light and dark themes
FAIR = "#a855f7"  # model fair value on the price chart
UP, DOWN = "#16a34a", "#dc2626"
FAMILIES = {
    "A_monetary": "Monetary",
    "B_network": "Network",
    "C_cashflow": "Cash flow",
    "D_multiples": "Multiples",
    "E_cost": "Cost basis",
    "F_shelf": "Other",
}
SECTORS = {
    "pow_monetary": "PoW money",
    "smart_contract_l1": "Smart-contract L1",
    "scaling": "Scaling",
    "defi": "DeFi",
    "oracle": "Oracle",
    "exchange": "Exchange",
    "metaverse": "Metaverse",
    "stablecoin": "Stablecoin",
}
ACRONYMS = {
    "mvrv", "mvtv", "nvt", "nvm", "dcf", "tvl", "qtm", "eth", "eoe", "aviv", "lppls", "fdv",
    "dfdv", "slrv", "rpv", "sthpl", "pv",
}  # fmt: skip
SMALL = {"to", "of", "and", "per", "on"}
RANGES = {"1M": 30, "3M": 91, "1Y": 365, "5Y": 1826, "All": None}

SNAPSHOT = pathlib.Path(__file__).resolve().parent.parent / "snapshot"


@st.cache_resource(show_spinner="Loading the published data...")
def _from_snapshot(stamp: int) -> str:
    db = str(pathlib.Path(tempfile.gettempdir()) / f"cryptoval-{stamp}.duckdb")
    snapshot.build(str(SNAPSHOT), db)
    return db


def _db_path() -> str:
    # The hosted app (Phase 6) has no data root, only the Parquet the daily run publishes.
    if "CRYPTOVAL_DATA_ROOT" not in os.environ and SNAPSHOT.is_dir():
        return _from_snapshot(snapshot.stamp(str(SNAPSHOT)))
    return str(pathlib.Path(os.environ.get("CRYPTOVAL_DATA_ROOT", "data")) / "cryptoval.duckdb")


@st.cache_data(ttl=600, show_spinner=False)
def _read(db: str, name: str, *args):
    con = duckdb.connect(db, read_only=True)
    try:
        return getattr(q, name)(con, *args)
    finally:
        con.close()


def _label(method_id: str) -> str:
    """mvrv_z_score -> MVRV Z Score."""
    words = method_id.split("_")
    return " ".join(
        w.upper() if w in ACRONYMS else w if i and w in SMALL else w.capitalize()
        for i, w in enumerate(words)
    )


def _money(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= div:
            return f"${v / div:,.2f}{unit}"
    return f"${v:,.2f}" if abs(v) >= 1 else f"${v:,.4f}"


def _freshness(db: str) -> None:
    fresh = _read(db, "freshness", dt.date.today())
    if fresh["stale"]:
        age = "empty" if fresh["panel_lag_days"] is None else f"{fresh['panel_lag_days']} days old"
        st.warning(f"The data is {age} (through {fresh['panel_through']}): run `make all`.")
    coin = fresh["coin_age_through"] or "not extracted"
    st.caption(
        f"{DISCLAIMER} Prices through {fresh['panel_through']}; BTC coin-age data through {coin}."
    )


def _colour_signed(v) -> str:
    return "" if pd.isna(v) or v == 0 else f"color: {UP if v > 0 else DOWN}"


# ---------------------------------------------------------------------------------- screener


def _screener(db: str) -> None:
    st.title("Cryptoasset valuation")
    table = _read(db, "overview")
    if table.empty:
        st.info("No asset has a price yet.")
        return
    priced = table.dropna(subset=["fair_value"])
    m = st.columns(3)
    m[0].metric("Tracked market cap", _money(table["market_cap_usd"].sum()), border=True)
    m[1].metric(
        "Trading below model fair value",
        f"{int((priced['median_upside'] > 0).sum())} of {len(priced)}",
        border=True,
        help="Assets whose price is below the median of their models' fair values.",
    )
    m[2].metric("Assets", len(table), border=True)

    sectors = [SECTORS.get(s, s) for s in dict.fromkeys(table["sector"].dropna())]
    pick = st.pills("Sector", sectors, label_visibility="collapsed")
    if pick:
        table = table[table["sector"].map(lambda s: SECTORS.get(s, s)) == pick]
    table = table.reset_index(drop=True).assign(
        sector=lambda t: t["sector"].map(lambda s: SECTORS.get(s, s)),
        agree=lambda t: (t["above"] / t["methods"]).where(t["methods"] > 0),
    )
    st.caption("Select a row to open that asset.")
    styled = table.style.map(
        _colour_signed, subset=["change_7d", "change_30d", "median_upside"]
    ).format({"fair_value": "${:,.2f}", "median_upside": "{:+.1%}"})
    num = st.column_config.NumberColumn
    event = st.dataframe(
        styled,
        key="screener",
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        on_select="rerun",
        selection_mode="single-row",
        column_order=[
            "symbol",
            "sector",
            "price_usd",
            "change_7d",
            "change_30d",
            "market_cap_usd",
            "last_90d",
            "fair_value",
            "median_upside",
            "agree",
            "methods",
        ],  # fmt: skip
        column_config={
            "symbol": st.column_config.TextColumn("Asset", pinned=True),
            "sector": "Sector",
            "price_usd": num("Price", format="dollar"),
            "change_7d": num("7d", format="percent"),
            "change_30d": num("30d", format="percent"),
            "market_cap_usd": num("Market cap", format="compact"),
            "last_90d": st.column_config.LineChartColumn("Last 90 days"),
            "fair_value": num(
                "Model fair value", help="Median of the asset's models' latest fair values."
            ),
            "median_upside": num("Upside", help="Model fair value against today's price."),
            "agree": st.column_config.ProgressColumn(
                "Models above price",
                format="percent",
                min_value=0.0,
                max_value=1.0,
                help="Share of the asset's models whose fair value is above today's price.",
            ),
            "methods": num("Models", help="How many valuation models apply to this asset."),
        },
    )
    if event.selection.rows:
        st.query_params["asset"] = table["symbol"].iloc[event.selection.rows[0]]
        st.rerun()
    with st.expander("Compare every model across every asset"):
        _agreement(db)


# ------------------------------------------------------------------------------------- asset


def _asset(db: str, symbol: str) -> None:
    table = _read(db, "overview")
    if symbol not in set(table["symbol"]):
        st.warning(f"No asset called {symbol!r}.")
        if st.button("← All assets"):
            _home()
        return
    row = table.set_index("symbol").loc[symbol]
    if st.button("← All assets"):
        _home()
    st.title(symbol)
    st.caption(SECTORS.get(row["sector"], row["sector"] or ""))
    m = st.columns(4)
    m[0].metric(
        "Price",
        _money(row["price_usd"]),
        None if pd.isna(row["change_7d"]) else f"{row['change_7d']:+.1%} 7d",
        border=True,
    )
    m[1].metric("Market cap", _money(row["market_cap_usd"]), border=True)
    m[2].metric(
        "Model fair value",
        _money(row["fair_value"]),
        None if pd.isna(row["median_upside"]) else f"{row['median_upside']:+.1%} vs price",
        border=True,
        help="Median of this asset's models' latest fair values.",
    )
    m[3].metric(
        "Models above price",
        f"{row['above']} of {row['methods']}" if row["methods"] else "none apply",
        border=True,
    )
    if isinstance(row["coverage_note"], str):
        st.caption(f"Data note: {row['coverage_note']}")
    _price_chart(db, symbol, row["fair_value"])
    st.subheader("Valuation models")
    _models(db, symbol, row["price_usd"])
    if symbol == "BTC":
        st.subheader("On-chain cycle indicators")
        _indicators(db)
    if not _read(db, "dcf_inputs").query("symbol == @symbol").empty:
        st.subheader("DCF calculator")
        _dcf(db, symbol)


def _home() -> None:
    st.query_params.clear()
    st.session_state.pop("screener", None)  # else the old row selection reopens the asset
    st.rerun()


def _price_chart(db: str, symbol: str, fair_value: float) -> None:
    prices = _read(db, "price_series", symbol)
    if prices.empty:
        return
    span = st.segmented_control("Range", list(RANGES), default="1Y", label_visibility="collapsed")
    days = RANGES.get(span or "All")
    if days:
        prices = prices[prices["metric_date"] > prices["metric_date"].iloc[-1] - dt.timedelta(days)]
    base = alt.Chart(prices).encode(
        x=alt.X("metric_date:T", title=None),
        y=alt.Y("price_usd:Q", title="Price, USD", scale=alt.Scale(zero=False)),
    )
    layers = [
        base.mark_line().encode(
            tooltip=[
                alt.Tooltip("metric_date:T", title="date"),
                alt.Tooltip("price_usd:Q", title="price", format=",.2f"),
            ]
        )
    ]
    if not pd.isna(fair_value):
        layers.append(
            alt.Chart(pd.DataFrame({"y": [fair_value]}))
            .mark_rule(color=FAIR, strokeDash=[6, 3], strokeWidth=2)
            .encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="model fair value", format=",.2f")])
        )
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    if not pd.isna(fair_value):
        st.caption("Purple dashed line: model fair value today.")


def _models(db: str, symbol: str, price: float) -> None:
    field = _read(db, "football_field", symbol)
    if field.empty:
        st.info(f"No valuation model produces a fair value for {symbol}.")
    else:
        field = field.assign(
            model=field["method_id"].map(_label),
            family=field["family"].map(lambda f: FAMILIES.get(f, f)),
            vs_price=field["mid"] / price - 1,
        )
        order = field["model"].tolist()  # cheapest fair value first
        y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        x = alt.X("low:Q", title="Fair value, USD (log scale)", scale=alt.Scale(type="log"))
        colour = alt.Color("family:N", title="Model family", scale=alt.Scale(scheme="tableau10"))
        tips = [
            alt.Tooltip("model:N"),
            alt.Tooltip("family:N"),
            alt.Tooltip("mid:Q", title="fair value", format=",.2f"),
            alt.Tooltip("low:Q", format=",.2f"),
            alt.Tooltip("high:Q", format=",.2f"),
            alt.Tooltip("vs_price:Q", title="vs price", format="+.0%"),
            alt.Tooltip("metric_date:T", title="as of"),
        ]
        whisker = alt.Chart(field).mark_rule(strokeWidth=3, opacity=0.6)
        dot = alt.Chart(field).mark_circle(size=110, opacity=1)
        layers = [
            whisker.encode(x=x, x2="high:Q", y=y, color=colour, tooltip=tips),
            dot.encode(x=alt.X("mid:Q"), y=y, color=colour, tooltip=tips),
            alt.Chart(pd.DataFrame({"price": [price]}))
            .mark_rule(color=PRICE, strokeDash=[6, 3], strokeWidth=2)
            .encode(x="price:Q", tooltip=[alt.Tooltip("price:Q", format=",.2f")]),
        ]
        st.altair_chart(alt.layer(*layers).properties(height=28 * len(field) + 50), width="stretch")
        st.caption(
            "Dot: each model's fair value. Line: its range across the model's assumptions. "
            "Orange dashed line: today's price. Models right of it say the asset is worth more."
        )
        with st.expander("All models as a table"):
            st.dataframe(
                field[["model", "family", "mid", "vs_price", "low", "high", "metric_date"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "model": "Model",
                    "family": "Family",
                    "mid": st.column_config.NumberColumn("Fair value", format="dollar"),
                    "vs_price": st.column_config.NumberColumn("vs price", format="percent"),
                    "low": st.column_config.NumberColumn("Low", format="dollar"),
                    "high": st.column_config.NumberColumn("High", format="dollar"),
                    "metric_date": st.column_config.DateColumn("As of"),
                },
            )
    refused = _read(db, "refusals", symbol)
    if not refused.empty:
        with st.expander(f"{len(refused)} models never apply to {symbol}"):
            st.dataframe(
                refused.assign(method_id=refused["method_id"].map(_label)),
                hide_index=True,
                width="stretch",
                column_config={"method_id": "Model", "reason": "Why not"},
            )


# ---------------------------------------------------------------------------------- research


def _indicators(db: str) -> None:
    name = st.selectbox("Indicator", q.INDICATORS, format_func=_label)
    series = _read(db, "indicator_series", name)
    if series.empty:
        st.info(f"No {_label(name)} history yet.")
        return
    chart = (
        alt.Chart(series)
        .mark_line()
        .encode(x=alt.X("metric_date:T", title=None), y=alt.Y("value:Q", title=_label(name)))
        .interactive(bind_y=False)
    )
    layers = [chart]
    for level in q.THRESHOLDS.get(name, []):
        layers.append(
            alt.Chart(pd.DataFrame({"y": [level]}))
            .mark_rule(color=PRICE, strokeDash=[6, 3])
            .encode(y="y:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")
    note = q.THRESHOLDS.get(name)
    st.caption(
        f"Latest {series['value'].iloc[-1]:.4g} on {series['metric_date'].iloc[-1]}. "
        "Scroll to zoom, drag to pan, double-click to reset."
        + (f" Dashed: the dissertation's stated thresholds {note}." if note else "")
    )


def _agreement(db: str) -> None:
    grid = _read(db, "agreement")
    if grid.empty:
        st.info("No fair values to compare.")
        return
    grid = grid.assign(
        shown=grid["log_ratio"].clip(-CLIP, CLIP), model=grid["method_id"].map(_label)
    )
    chart = (
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("model:N", title=None),
            y=alt.Y("symbol:N", title=None),
            color=alt.Color(
                "shown:Q",
                title="log10(fair / price)",
                scale=alt.Scale(scheme="redblue", domain=[-CLIP, CLIP], reverse=True),
            ),
            tooltip=[
                "symbol",
                "model",
                "metric_date",
                alt.Tooltip("log_ratio:Q", format=".2f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Blue: model says undervalued; red: overvalued. Colour clipped at ±{CLIP:g} orders of "
        f"magnitude (raw range {grid['log_ratio'].min():.1f} to {grid['log_ratio'].max():.1f}). "
        "Each cell is the model's grid median on its latest applicable day, against that "
        "day's price."
    )


def _dcf(db: str, symbol: str) -> None:
    rows = _read(db, "dcf_inputs").query("symbol == @symbol").reset_index(drop=True)
    choice = st.selectbox("Model", rows["method_id"].tolist(), format_func=_label)
    row = rows[rows["method_id"] == choice].iloc[0]
    clamped: list[str] = []

    def start(label: str, value: float, lo: float, hi: float) -> float:
        # A stored setting outside the slider's range would make st.slider raise.
        if not lo <= value <= hi:
            clamped.append(f"{label} {value:g}")
        return min(max(value, lo), hi)

    growth = list(row["growth"])
    cols = st.columns(2)
    discount = cols[0].slider(
        "Discount rate",
        0.01,
        0.40,
        start("discount rate", float(row["discount_rate"]), 0.01, 0.40),
        0.01,
    )
    terminal = cols[1].slider(
        "Terminal multiple (0 = Gordon growth)",
        0.0,
        40.0,
        start("terminal multiple", float(row["terminal_multiple"] or 0.0), 0.0, 40.0),
        1.0,
    )
    g = st.columns(3)
    growth = [
        g[i].slider(
            f"Stage {i + 1} growth",
            -0.20,
            0.60,
            start(f"stage {i + 1} growth", float(growth[i]), -0.20, 0.60),
            0.01,
        )
        for i in range(3)
    ]
    if clamped:
        st.caption(
            f"The stored {', '.join(clamped)} is outside the slider range, so the slider starts "
            "at its nearest end and these numbers will not match the stored estimate."
        )
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
        f"from the {_label(row['method_id'])} row of {row['metric_date']}. "
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
    db = _db_path()
    if not pathlib.Path(db).exists():
        st.title("Cryptoasset valuation")
        st.error(f"No database at {db}. Build it with `make all`, then reload.")
        return
    symbol = st.query_params.get("asset")
    if symbol:
        _asset(db, symbol.upper())
    else:
        _screener(db)
    _freshness(db)


main()
