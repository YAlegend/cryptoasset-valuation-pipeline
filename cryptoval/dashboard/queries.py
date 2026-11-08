"""Read-only queries behind the dashboard (spec 12 deliverable 3, Phase 5).

Every function takes an open DuckDB connection and returns plain data; the app owns the
connection's lifetime. Signals (dim_method.returns_fair_value = false) are filtered by the
method's DECLARATION, never by whether fair_value happens to be null, so a signal can never be
drawn as a price.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from cryptoval.methods.cashflow import three_stage_pv

STALE_DAYS = 3
DCF_METHODS = ("defi_net_revenue_dcf", "eth_three_stage_dcf")
SIGNALS = ("slrv", "rpv", "sthpl", "aviv")
INDICATORS = ("mvrv", "mayer_multiple", "puell_multiple", "liveliness", *SIGNALS)
# The dissertation's own stated thresholds (sections 5.5, 6.1, 6.2); none is drawn where it
# stated none.
THRESHOLDS: dict[str, list[float]] = {"slrv": [0.04], "rpv": [0.02, 0.00001], "sthpl": [1.0]}

# Latest applicable date per (symbol, method), over price-returning methods only.
_LATEST_FAIR_VALUES = """
with fv as (
    select e.symbol, e.method_id, cast(e.metric_date as date) as metric_date, e.fair_value,
           e.is_circular
    from fct_valuation_estimate e
    join dim_method m using (method_id)
    where m.returns_fair_value and e.applicable and e.fair_value > 0 {where}
),
latest as (
    select symbol, method_id, max(metric_date) as metric_date from fv group by all
)
select fv.* from fv join latest using (symbol, method_id, metric_date)
"""


def _date(value) -> dt.date | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).date()


def freshness(con, today: dt.date) -> dict:
    through, vintage = con.execute(
        "select max(metric_date), max(panel_vintage_ts) from fct_asset_metrics_daily"
    ).fetchone()
    try:
        coin = con.execute("select max(metric_date) from fct_btc_cointime_daily").fetchone()[0]
    except Exception:  # the coin-age mart is optional until Plan 3b's extraction has run
        coin = None
    through = _date(through)
    lag = None if through is None else (today - through).days  # None: a built but empty panel
    return {
        "panel_through": through,
        "coin_age_through": _date(coin),
        "panel_vintage": vintage,
        "panel_lag_days": lag,
        "stale": lag is None or lag > STALE_DAYS,
    }


def latest_price(con, symbol: str) -> tuple[dt.date, float] | None:
    row = con.execute(
        "select metric_date, price_usd from fct_asset_metrics_daily "
        "where symbol = ? and price_usd > 0 order by metric_date desc limit 1",
        [symbol],
    ).fetchone()
    return None if row is None else (_date(row[0]), float(row[1]))


def football_field(con, symbol: str) -> pd.DataFrame:
    sql = f"""
    select f.method_id, m.family, f.metric_date, min(f.fair_value) as low,
           max(f.fair_value) as high, median(f.fair_value) as mid, count(*) as n_cells,
           bool_or(f.is_circular) as is_circular
    from ({_LATEST_FAIR_VALUES.format(where="and e.symbol = ?")}) f
    join dim_method m using (method_id)
    group by all
    order by mid
    """
    out = con.execute(sql, [symbol]).fetchdf()
    out["metric_date"] = out["metric_date"].map(_date)
    return out


def refusals(con, symbol: str) -> pd.DataFrame:
    """Methods that never apply to this asset, each with its latest reason."""
    return con.execute(
        "select method_id, arg_max(reason, metric_date) as reason "
        "from fct_valuation_estimate where symbol = ? "
        "group by method_id having not bool_or(applicable) order by method_id",
        [symbol],
    ).fetchdf()


def indicator_series(con, name: str, symbol: str = "BTC") -> pd.DataFrame:
    if name not in INDICATORS:
        raise ValueError(f"unknown indicator {name!r}; expected one of {INDICATORS}")
    if name in SIGNALS:
        raw = con.execute(
            "select cast(metric_date as date) as metric_date, diagnostics "
            "from fct_valuation_estimate where symbol = ? and method_id = ? and applicable "
            "order by metric_date",
            [symbol, name],
        ).fetchdf()
        values = raw["diagnostics"].map(lambda d: json.loads(d).get(name))
        out = pd.DataFrame({"metric_date": raw["metric_date"], "value": values})
    elif name == "liveliness":
        out = con.execute(
            "select metric_date, liveliness as value from fct_btc_cointime_daily "
            "where symbol = ? order by metric_date",
            [symbol],
        ).fetchdf()
    else:
        panel = con.execute(
            "select metric_date, mvrv, price_usd, issuance_usd from fct_asset_metrics_daily "
            "where symbol = ? order by metric_date",
            [symbol],
        ).fetchdf()
        if name == "mvrv":
            value = panel["mvrv"]
        elif name == "mayer_multiple":
            value = panel["price_usd"] / panel["price_usd"].rolling(200).mean()
        else:  # puell_multiple
            value = panel["issuance_usd"] / panel["issuance_usd"].rolling(365).mean()
        out = pd.DataFrame({"metric_date": panel["metric_date"], "value": value})
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["metric_date"] = out["metric_date"].map(_date)
    out["value"] = out["value"].astype(float)
    return out


def agreement(con) -> pd.DataFrame:
    """log10(median fair value / that day's price) per (symbol, method), latest applicable day."""
    sql = f"""
    with cell as (
        select symbol, method_id, metric_date, median(fair_value) as fair_value
        from ({_LATEST_FAIR_VALUES.format(where="")}) group by all
    )
    select c.symbol, c.method_id, c.metric_date, log10(c.fair_value / p.price_usd) as log_ratio
    from cell c
    join fct_asset_metrics_daily p on p.symbol = c.symbol and p.metric_date = c.metric_date
    where p.price_usd > 0
    order by c.symbol, c.method_id
    """
    out = con.execute(sql).fetchdf()
    out["metric_date"] = out["metric_date"].map(_date)
    return out


def overview(con) -> pd.DataFrame:
    """One row per asset, largest market cap first: sector, latest price, 7d and 30d change,
    market cap, the last 90 days of price, and how the price-returning methods' latest fair
    values (as in agreement) sit against price."""
    out = con.execute(
        """
        with p as (select symbol, metric_date, price_usd
                   from fct_asset_metrics_daily where price_usd > 0),
        last as (select symbol, max(metric_date) as metric_date from p group by symbol),
        known as (
            select symbol,
                   arg_max(supply, metric_date) filter (where supply > 0) as supply,
                   max(metric_date) filter (where supply > 0) as supply_date,
                   arg_max(market_cap_usd, metric_date) filter (where market_cap_usd > 0) as cap,
                   max(metric_date) filter (where market_cap_usd > 0) as cap_date
            from fct_asset_metrics_daily group by symbol
        ),
        spark as (
            select p.symbol, list(p.price_usd order by p.metric_date) as last_90d
            from p join last l on p.symbol = l.symbol and p.metric_date > l.metric_date - 90
            group by p.symbol
        )
        select l.symbol, a.sector, a.coverage_note, l.metric_date as price_date, n.price_usd,
               -- supply lands a day after price and stops years early for BNB and DOT, so a
               -- month-stale panel gives way to CoinGecko's current total supply
               coalesce(
                   case when k.supply_date >= l.metric_date - 30 then n.price_usd * k.supply end,
                   case when k.cap_date >= l.metric_date - 30 then k.cap end,
                   n.price_usd * a.total_supply
               ) as market_cap_usd,
               n.price_usd / w.price_usd - 1 as change_7d,
               n.price_usd / m.price_usd - 1 as change_30d,
               s.last_90d
        from last l
        join p n using (symbol, metric_date)
        left join known k on k.symbol = l.symbol
        left join dim_asset a on a.symbol = l.symbol
        left join spark s on s.symbol = l.symbol
        left join p w on w.symbol = l.symbol and w.metric_date = l.metric_date - 7
        left join p m on m.symbol = l.symbol and m.metric_date = l.metric_date - 30
        order by market_cap_usd desc nulls last, l.symbol
        """
    ).fetchdf()
    votes = (
        agreement(con)
        .groupby("symbol")["log_ratio"]
        .agg(methods="size", above=lambda r: int((r > 0).sum()), median_log_ratio="median")
        .reset_index()
    )
    out = out.merge(votes, on="symbol", how="left")
    out["methods"] = out["methods"].fillna(0).astype(int)
    out["above"] = out["above"].fillna(0).astype(int)
    out["median_upside"] = 10 ** out.pop("median_log_ratio") - 1
    out["fair_value"] = out["price_usd"] * (1 + out["median_upside"])
    out["price_date"] = out["price_date"].map(_date)
    return out


def price_series(con, symbol: str) -> pd.DataFrame:
    out = con.execute(
        "select metric_date, price_usd from fct_asset_metrics_daily "
        "where symbol = ? and price_usd > 0 order by metric_date",
        [symbol],
    ).fetchdf()
    out["metric_date"] = out["metric_date"].map(_date)
    return out


def fair_value_series(con, symbol: str, families: list[str]) -> pd.DataFrame:
    """Daily model fair value: the median, across the given families' models applicable that
    day, of each model's grid median. Geometric (a median of logs), as the overview's is."""
    out = con.execute(
        """
        with cell as (
            select cast(e.metric_date as date) as metric_date, e.method_id,
                   median(e.fair_value) as fair_value
            from fct_valuation_estimate e
            join dim_method m using (method_id)
            where e.symbol = ? and m.returns_fair_value and e.applicable and e.fair_value > 0
              and list_contains(?, m.family)
            group by all
        )
        select metric_date, exp(median(ln(fair_value))) as fair_value, count(*) as models
        from cell group by metric_date order by metric_date
        """,
        [symbol, families],
    ).fetchdf()
    out["metric_date"] = out["metric_date"].map(_date)
    return out


def model_series(con, symbol: str, method_id: str) -> pd.DataFrame:
    """One model's grid range and median per applicable day, beside that day's price."""
    out = con.execute(
        """
        select cast(e.metric_date as date) as metric_date, min(e.fair_value) as low,
               median(e.fair_value) as mid, max(e.fair_value) as high,
               any_value(p.price_usd) as price_usd
        from fct_valuation_estimate e
        left join fct_asset_metrics_daily p
               on p.symbol = e.symbol and p.metric_date = cast(e.metric_date as date)
        where e.symbol = ? and e.method_id = ? and e.applicable and e.fair_value > 0
        group by 1 order by 1
        """,
        [symbol, method_id],
    ).fetchdf()
    out["metric_date"] = out["metric_date"].map(_date)
    return out


def method_info(con, method_id: str) -> dict | None:
    row = con.execute(
        "select family, requires, citation from dim_method where method_id = ?", [method_id]
    ).fetchone()
    return None if row is None else dict(zip(("family", "requires", "citation"), row, strict=True))


def dcf_inputs(con) -> pd.DataFrame:
    """The recorded inputs of each DCF's latest applicable row, one per (symbol, method)."""
    placeholders = ", ".join("?" for _ in DCF_METHODS)
    raw = con.execute(
        f"""
        select symbol, method_id, cast(metric_date as date) as metric_date,
               arg_max(inputs_used, fair_value) as inputs_used
        from fct_valuation_estimate
        where applicable and method_id in ({placeholders})
          and (symbol, method_id, metric_date) in (
              select (symbol, method_id, max(metric_date)) from fct_valuation_estimate
              where applicable and method_id in ({placeholders}) group by symbol, method_id
          )
        group by all
        order by symbol, method_id
        """,
        [*DCF_METHODS, *DCF_METHODS],
    ).fetchdf()
    fields = [
        "annual_cash_flow_usd",
        "supply",
        "discount_rate",
        "growth",
        "stage_years",
        "terminal_multiple",
    ]
    parsed = raw["inputs_used"].map(json.loads)
    out = raw[["symbol", "method_id", "metric_date"]].copy()
    for name in fields:
        out[name] = parsed.map(lambda d, n=name: d.get(n))
    out["metric_date"] = out["metric_date"].map(_date)
    return out


def dcf_value(
    cash_flow: float,
    supply: float,
    growth: list[float],
    discount: float,
    years: list[int],
    terminal_multiple: float | None,
) -> tuple[float, float]:
    """Fair value per token and the terminal share; three_stage_pv's ValueError propagates."""
    present_value, terminal_share = three_stage_pv(
        cash_flow, tuple(growth), discount, tuple(years), terminal_multiple
    )
    return present_value / supply, terminal_share
