"""A miniature of the marts the dashboard reads, shared by the query and app tests."""

import datetime as dt
import json

import pandas as pd

DAY0 = dt.date(2025, 1, 1)
N = 400
LAST = DAY0 + dt.timedelta(days=N - 1)
PREV = LAST - dt.timedelta(days=1)
DCF_INPUTS = {
    "annual_cash_flow_usd": 1e6,
    "supply": 1e4,
    "discount_rate": 0.2,
    "growth": [0.1, 0.05, 0.02],
    "stage_years": [3, 3, 4],
    "terminal_multiple": 10.0,
}


def _estimate(
    symbol,
    day,
    method_id,
    fair_value,
    applicable,
    *,
    circular=False,
    reason=None,
    inputs=None,
    diagnostics=None,
):
    return {
        "symbol": symbol,
        "metric_date": pd.Timestamp(day),
        "method_id": method_id,
        "fair_value": fair_value,
        "applicable": applicable,
        "reason": reason,
        "is_circular": circular,
        "inputs_used": json.dumps(inputs or {}),
        "diagnostics": json.dumps(diagnostics or {}),
    }


def build(con) -> None:
    days = [DAY0 + dt.timedelta(days=i) for i in range(N)]
    vintage = pd.Timestamp("2026-02-05 12:00", tz="UTC")
    btc = pd.DataFrame(
        {
            "symbol": "BTC",
            "metric_date": days,
            "price_usd": [100.0 + i for i in range(N)],
            "mvrv": 2.0,
            "issuance_usd": 10.0,
            "panel_vintage_ts": vintage,
        }
    )
    usdc = btc.assign(symbol="USDC", price_usd=1.0, mvrv=None, issuance_usd=None)
    panel = pd.concat([btc, usdc], ignore_index=True)
    cointime = pd.DataFrame({"symbol": "BTC", "metric_date": days, "liveliness": 0.6})
    methods = pd.DataFrame(
        [
            ("realized_price", "E_cost", True),
            ("mayer_multiple", "E_cost", True),
            ("slrv", "E_cost", False),
            ("defi_net_revenue_dcf", "C_cashflow", True),
            ("never_applies", "D_multiples", True),
        ],
        columns=["method_id", "family", "returns_fair_value"],
    )
    estimates = pd.DataFrame(
        [
            _estimate("BTC", PREV, "realized_price", 50.0, True, circular=True),
            _estimate("BTC", LAST, "realized_price", 60.0, True, circular=True),
            _estimate("BTC", PREV, "defi_net_revenue_dcf", 40.0, True, inputs=DCF_INPUTS),
            _estimate("BTC", PREV, "defi_net_revenue_dcf", 80.0, True, inputs=DCF_INPUTS),
            _estimate("BTC", LAST, "defi_net_revenue_dcf", None, False, reason="no supply"),
            # a signal with a (wrong) non-null fair value: the filter must be the method's
            # declaration, not the column's nullness
            _estimate("BTC", LAST, "slrv", 999.0, True, diagnostics={"slrv": 0.05}),
            _estimate("BTC", LAST, "never_applies", None, False, reason="nope"),
            _estimate("USDC", LAST, "never_applies", None, False, reason="nope"),
            _estimate("USDC", LAST, "realized_price", None, False, reason="no mvrv"),
        ]
    )
    for name, frame in {
        "fct_asset_metrics_daily": panel,
        "fct_btc_cointime_daily": cointime,
        "dim_method": methods,
        "fct_valuation_estimate": estimates,
    }.items():
        con.register("_frame", frame)
        con.execute(f"create or replace table {name} as select * from _frame")
        con.unregister("_frame")
    con.execute("alter table fct_asset_metrics_daily alter metric_date type date")
    con.execute("alter table fct_btc_cointime_daily alter metric_date type date")
