import json

import numpy as np
import pandas as pd
import pytest

from cryptoval.methods.base import Estimate, MethodSpec
from cryptoval.methods.registry import (
    ESTIMATE_COLUMNS,
    load_method_specs,
    param_set_hash,
    run_methods,
)


class DoublePrice:
    """A stub method: fair value is twice the realized price."""

    spec = MethodSpec(
        method_id="double_realized",
        family="E_cost",
        requires=("realized_cap", "supply"),
        min_density=0.5,
        min_rows=3,
        params={"multiple": 2.0},
        citation="test",
    )

    def estimate(self, frame, params):
        out = []
        for _, row in frame.iterrows():
            out.append(
                Estimate(
                    symbol=row["symbol"],
                    metric_date=row["metric_date"],
                    method_id=self.spec.method_id,
                    param_set_hash=param_set_hash(params),
                    fair_value=params["multiple"] * row["realized_cap"] / row["supply"],
                    band_low=None,
                    band_high=None,
                    inputs_used={"realized_cap": row["realized_cap"], "supply": row["supply"]},
                    diagnostics={},
                    applicable=True,
                    panel_vintage_ts=row["panel_vintage_ts"],
                )
            )
        return out


def _panel(symbols=("BTC", "BNB")) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        n = 10
        frame = pd.DataFrame(
            {
                "symbol": [sym] * n,
                "metric_date": pd.date_range("2026-01-01", periods=n, freq="D").date,
                "realized_cap": np.linspace(1e12, 1.1e12, n),
                "supply": np.full(n, 1.9e7),
                "panel_vintage_ts": pd.Timestamp("2026-10-12T08:00:00Z"),
            }
        )
        if sym == "BNB":
            # 20% dense, like the real BNB market cap: below the method's floor.
            frame.loc[2:, "realized_cap"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_estimates_are_produced_per_symbol_and_date():
    out = run_methods(_panel(("BTC",)), {"double_realized": DoublePrice()})
    assert len(out) == 10
    assert list(out.columns) == ESTIMATE_COLUMNS
    assert out["fair_value"].iloc[0] == pytest.approx(2.0 * 1e12 / 1.9e7)


def test_an_inapplicable_asset_yields_one_refusal_row_not_a_silent_gap():
    out = run_methods(_panel(), {"double_realized": DoublePrice()})
    bnb = out[out["symbol"] == "BNB"]
    # One row, not ten, and not zero: a refusal is a finding and must be queryable.
    assert len(bnb) == 1
    assert not bool(bnb["applicable"].iloc[0])
    assert bnb["fair_value"].isna().all()
    assert "20.0% dense" in bnb["reason"].iloc[0]


def test_the_panel_vintage_travels_onto_every_estimate():
    """Review Focus 4: quarantine can change a velocity between runs, so an estimate must
    name the panel state it was computed from."""
    out = run_methods(_panel(("BTC",)), {"double_realized": DoublePrice()})
    assert (out["panel_vintage_ts"] == pd.Timestamp("2026-10-12T08:00:00Z")).all()


def test_circular_inputs_are_recorded_on_every_row():
    class CircularMethod(DoublePrice):
        spec = MethodSpec(
            method_id="circular_stub",
            family="A_monetary",
            requires=("realized_cap", "supply", "velocity_total"),
            min_density=0.0,
            min_rows=1,
            params={"multiple": 2.0},
            citation="test",
        )

    panel = _panel(("BTC",))
    panel["velocity_total"] = 0.05
    out = run_methods(panel, {"circular_stub": CircularMethod()})
    # realized_cap AND velocity_total are both market-cap-derived.
    assert out["circular_inputs"].iloc[0] == "realized_cap|velocity_total"


def test_the_same_params_hash_the_same_and_different_params_do_not():
    assert param_set_hash({"a": 1, "b": 2}) == param_set_hash({"b": 2, "a": 1})
    assert param_set_hash({"a": 1}) != param_set_hash({"a": 2})


def test_inputs_and_diagnostics_are_stored_as_json_strings():
    out = run_methods(_panel(("BTC",)), {"double_realized": DoublePrice()})
    # Parquet has no dict column type worth relying on, and JSON keeps the row readable
    # from SQL via DuckDB's json functions.
    parsed = json.loads(out["inputs_used"].iloc[0])
    assert parsed["supply"] == 1.9e7


def test_a_method_that_raises_produces_a_refusal_rather_than_killing_the_run():
    class Exploding(DoublePrice):
        spec = MethodSpec(
            method_id="boom",
            family="E_cost",
            requires=("realized_cap", "supply"),
            min_density=0.0,
            min_rows=1,
            params={},
            citation="test",
        )

        def estimate(self, frame, params):
            raise ZeroDivisionError("supply was zero")

    out = run_methods(_panel(("BTC",)), {"boom": Exploding()})
    assert len(out) == 1
    assert not bool(out["applicable"].iloc[0])
    assert "ZeroDivisionError" in out["reason"].iloc[0]
    assert "supply was zero" in out["reason"].iloc[0]


def test_method_specs_load_from_yaml():
    specs = load_method_specs()
    assert "realized_price" in specs
    spec = specs["realized_price"]
    assert spec.family == "E_cost"
    assert "realized_cap" in spec.requires
    assert 0.0 <= spec.min_density <= 1.0
    assert spec.citation
