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
    "dfdv", "slrv", "rpv", "sthpl", "pv", "usd",
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


# ---------------------------------------------------------------------------------- charts


def _lines(data: pd.DataFrame, y_title: str, fmt: str = ",.2f", colours=None) -> alt.LayerChart:
    """Long data (metric_date, series, value) as lines with a hover crosshair that reads every
    series at the pointer's date; scroll or drag the x axis to zoom, double-click to reset."""
    hover = alt.selection_point(
        fields=["metric_date"], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    scale = alt.Scale(domain=list(colours), range=list(colours.values())) if colours else None
    base = alt.Chart(data).encode(
        x=alt.X("metric_date:T", title=None),
        y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color(
            "series:N", title=None, legend=alt.Legend(orient="top"), scale=scale or alt.Undefined
        ),
    )
    shown = alt.when(hover)
    wide = data.pivot_table(index="metric_date", columns="series", values="value").reset_index()
    tips = [alt.Tooltip("metric_date:T", title="date")] + [
        alt.Tooltip(field=c, type="quantitative", format=fmt) for c in wide.columns[1:]
    ]
    return alt.layer(
        base.mark_line().add_params(zoom),
        base.mark_point(filled=True, size=60).encode(
            opacity=shown.then(alt.value(1)).otherwise(alt.value(0))
        ),
        alt.Chart(wide)
        .mark_rule(color="gray")
        .encode(
            x="metric_date:T",
            opacity=shown.then(alt.value(0.6)).otherwise(alt.value(0)),
            tooltip=tips,
        )
        .add_params(hover),
    )


def _span(prices: pd.DataFrame, key: str) -> pd.DataFrame:
    span = st.segmented_control(
        "Range", list(RANGES), default="1Y", key=key, label_visibility="collapsed"
    )
    days = RANGES.get(span or "All")
    if not days or prices.empty:
        return prices
    return prices[prices["metric_date"] > prices["metric_date"].iloc[-1] - dt.timedelta(days)]


def _columns() -> dict:
    num = st.column_config.NumberColumn
    return {
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
    }


def _styled(table: pd.DataFrame):
    table = table.reset_index(drop=True).assign(
        sector=lambda t: t["sector"].map(lambda s: SECTORS.get(s, s)),
        agree=lambda t: (t["above"] / t["methods"]).where(t["methods"] > 0),
    )
    return table, table.style.map(
        _colour_signed, subset=["change_7d", "change_30d", "median_upside"]
    ).format({"fair_value": "${:,.2f}", "median_upside": "{:+.1%}"})


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
    screener, compare = st.tabs(["Screener", "Compare"])
    with screener:
        _table(db, table)
    with compare:
        _compare(db, table)


def _table(db: str, table: pd.DataFrame) -> None:
    sectors = [SECTORS.get(s, s) for s in dict.fromkeys(table["sector"].dropna())]
    pick = st.pills("Sector", sectors, label_visibility="collapsed")
    if pick:
        table = table[table["sector"].map(lambda s: SECTORS.get(s, s)) == pick]
    table, styled = _styled(table)
    st.caption("Select a row to open that asset. Click a column header to sort.")
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
        column_config=_columns(),
    )
    if event.selection.rows:
        st.query_params["asset"] = table["symbol"].iloc[event.selection.rows[0]]
        st.rerun()
    with st.expander("Compare every model across every asset"):
        _agreement(db)


def _compare(db: str, table: pd.DataFrame) -> None:
    symbols = table["symbol"].tolist()
    picks = st.multiselect(
        "Assets to compare (up to 5)", symbols, default=symbols[:3], max_selections=5
    )
    if not picks:
        st.info("Pick an asset or two to compare.")
        return
    frames = []
    for symbol in picks:
        prices = _read(db, "price_series", symbol)
        if not prices.empty:
            frames.append(prices.assign(series=symbol))
    data = _span(pd.concat(frames), "compare_range") if frames else pd.DataFrame()
    if not data.empty:
        first = data.groupby("series")["price_usd"].transform("first")
        data = data.assign(value=data["price_usd"] / first * 100)
        st.altair_chart(
            _lines(data, "Price, indexed to 100", fmt=".1f").properties(height=360),
            width="stretch",
        )
        st.caption(
            "Each asset's price rebased to 100 at the start of the range. Hover to read every "
            "asset on a date; scroll to zoom, double-click to reset."
        )
    picked, styled = _styled(table.set_index("symbol").loc[picks].reset_index())
    upside = picked.dropna(subset=["median_upside"])
    if not upside.empty:
        st.altair_chart(
            alt.Chart(upside)
            .mark_bar()
            .encode(
                x=alt.X(
                    "median_upside:Q", title="Upside to model fair value", axis=alt.Axis(format="%")
                ),
                y=alt.Y("symbol:N", title=None, sort=picks),
                color=alt.when(alt.datum.median_upside > 0)
                .then(alt.value(UP))
                .otherwise(alt.value(DOWN)),
                tooltip=[
                    "symbol",
                    alt.Tooltip("fair_value:Q", title="model fair value", format=",.2f"),
                    alt.Tooltip("median_upside:Q", title="upside", format="+.1%"),
                ],
            )
            .properties(height=40 * len(upside) + 30),
            width="stretch",
        )
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        column_order=[
            "symbol",
            "price_usd",
            "change_7d",
            "change_30d",
            "market_cap_usd",
            "fair_value",
            "median_upside",
            "agree",
            "methods",
        ],  # fmt: skip
        column_config=_columns(),
    )


# ------------------------------------------------------------------------------------- asset


def _asset(db: str, symbol: str) -> None:
    table = _read(db, "overview")
    if symbol not in set(table["symbol"]):
        st.warning(f"No asset called {symbol!r}.")
        if st.button("← All assets"):
            _home()
        return
    row = table.set_index("symbol").loc[symbol]
    price = row["price_usd"]
    if st.button("← All assets"):
        _home()
    st.title(symbol)
    st.caption(SECTORS.get(row["sector"], row["sector"] or ""))

    field = _read(db, "football_field", symbol)
    families = [f for f in dict.fromkeys(field["family"])]
    picked = families
    if families:
        names = [FAMILIES.get(f, f) for f in families]
        chosen = st.pills(
            "Models included",
            names,
            selection_mode="multi",
            default=names,
            key="families",
            help="Turn model families off to see the fair value without them.",
        )
        picked = [f for f, n in zip(families, names, strict=True) if n in chosen]
    votes = _read(db, "agreement")
    votes = votes[votes["symbol"] == symbol].merge(field[["method_id", "family"]])
    votes = votes[votes["family"].isin(picked)]
    fair = price * 10 ** votes["log_ratio"].median() if len(votes) else float("nan")

    m = st.columns(4)
    m[0].metric(
        "Price",
        _money(price),
        None if pd.isna(row["change_7d"]) else f"{row['change_7d']:+.1%} 7d",
        border=True,
    )
    m[1].metric("Market cap", _money(row["market_cap_usd"]), border=True)
    m[2].metric(
        "Model fair value",
        _money(fair),
        None if pd.isna(fair) else f"{fair / price - 1:+.1%} vs price",
        border=True,
        help="Median of the included models' latest fair values.",
    )
    m[3].metric(
        "Models above price",
        f"{int((votes['log_ratio'] > 0).sum())} of {len(votes)}" if len(votes) else "none apply",
        border=True,
    )
    if isinstance(row["coverage_note"], str):
        st.caption(f"Data note: {row['coverage_note']}")
    _price_chart(db, symbol, picked)
    st.subheader("Valuation models")
    _models(db, symbol, price, field[field["family"].isin(picked)])
    model = st.query_params.get("model")
    if model and model in set(field["method_id"]):
        _model_detail(db, symbol, model)
    if symbol == "BTC":
        st.subheader("On-chain cycle indicators")
        _indicators(db)
    if not _read(db, "dcf_inputs").query("symbol == @symbol").empty:
        st.subheader("DCF calculator")
        _dcf(db, symbol)


def _home() -> None:
    st.query_params.clear()
    for key in ("screener", "models", "families"):  # else old selections reopen what was left
        st.session_state.pop(key, None)
    st.rerun()


def _price_chart(db: str, symbol: str, families: list[str]) -> None:
    prices = _read(db, "price_series", symbol)
    if prices.empty:
        return
    prices = _span(prices, "price_range")
    series = {"Price": prices.rename(columns={"price_usd": "value"})}
    if families:
        fair = _read(db, "fair_value_series", symbol, families)
        fair = fair[fair["metric_date"] >= prices["metric_date"].min()]
        series["Model fair value"] = fair.rename(columns={"fair_value": "value"})
    data = pd.concat(
        [f[["metric_date", "value"]].assign(series=name) for name, f in series.items()]
    )
    colours = {"Price": "#60a5fa", "Model fair value": FAIR}
    st.altair_chart(
        _lines(data, "USD", colours={k: colours[k] for k in series}).properties(height=340),
        width="stretch",
    )
    st.caption(
        "Model fair value: the median of the included models' fair values on each day. "
        "Hover to read both; scroll to zoom, double-click to reset."
    )


def _models(db: str, symbol: str, price: float, field: pd.DataFrame) -> None:
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
        click = alt.selection_point(name="model", fields=["method_id"], on="click")
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
        dot = alt.Chart(field).mark_circle(size=140, cursor="pointer")
        layers = [
            whisker.encode(x=x, x2="high:Q", y=y, color=colour, tooltip=tips),
            dot.encode(
                x=alt.X("mid:Q"),
                y=y,
                color=colour,
                tooltip=tips,
                opacity=alt.when(click).then(alt.value(1)).otherwise(alt.value(0.35)),
            ).add_params(click),
            alt.Chart(pd.DataFrame({"price": [price]}))
            .mark_rule(color=PRICE, strokeDash=[6, 3], strokeWidth=2)
            .encode(x="price:Q", tooltip=[alt.Tooltip("price:Q", format=",.2f")]),
        ]
        event = st.altair_chart(
            alt.layer(*layers).properties(height=28 * len(field) + 50),
            width="stretch",
            on_select="rerun",
            key="models",
        )
        clicked = [p["method_id"] for p in event.selection.get("model", []) if "method_id" in p]
        if clicked and clicked[0] != st.query_params.get("model"):
            st.query_params["model"] = clicked[0]
            st.rerun()
        st.caption(
            "Dot: each model's fair value. Line: its range across the model's assumptions. "
            "Orange dashed line: today's price; models right of it say the asset is worth more. "
            "**Click a dot to open that model.**"
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


def _model_detail(db: str, symbol: str, method_id: str) -> None:
    info = _read(db, "method_info", method_id) or {}
    history = _read(db, "model_series", symbol, method_id)
    with st.container(border=True):
        head = st.columns([5, 1])
        head[0].subheader(_label(method_id))
        if head[1].button("Close", key="close_model", width="stretch"):
            del st.query_params["model"]
            st.session_state.pop("models", None)
            st.rerun()
        family = FAMILIES.get(info.get("family"), info.get("family"))
        st.caption(f"{family} model. Source: {info.get('citation') or 'not recorded'}.")
        if info.get("requires"):
            inputs = ", ".join(_label(r) for r in info["requires"].split("|"))
            st.markdown(f"**Inputs:** {inputs}")
        if history.empty:
            return
        last = history.iloc[-1]
        m = st.columns(3)
        vs = (
            None
            if pd.isna(last["price_usd"])
            else f"{last['mid'] / last['price_usd'] - 1:+.1%} vs price"
        )
        m[0].metric("Fair value", _money(last["mid"]), vs, border=True)
        m[1].metric("Range", f"{_money(last['low'])} to {_money(last['high'])}", border=True)
        m[2].metric("As of", str(last["metric_date"]), border=True)
        history = _span(history, "model_range")
        data = pd.concat(
            [
                history[["metric_date", "price_usd"]]
                .rename(columns={"price_usd": "value"})
                .assign(series="Price"),
                history[["metric_date", "mid"]]
                .rename(columns={"mid": "value"})
                .assign(series="Fair value"),
            ]
        ).dropna()
        band = (
            alt.Chart(history)
            .mark_area(opacity=0.2, color=FAIR)
            .encode(x="metric_date:T", y="low:Q", y2="high:Q")
        )
        lines = _lines(data, "USD", colours={"Price": "#60a5fa", "Fair value": FAIR})
        st.altair_chart(alt.layer(band, lines).properties(height=300), width="stretch")
        st.caption("Shaded: the model's range across its assumptions on each day.")


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
