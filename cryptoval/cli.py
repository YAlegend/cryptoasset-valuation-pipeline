"""Command-line entry point: the only module that wires the pieces together."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import pathlib
import sys

import httpx
import pandas as pd

import cryptoval.methods  # noqa: F401 -- import registers every method family
import cryptoval.scoring  # noqa: F401 -- import registers every model family
from cryptoval.budget import ByteBudgetGate
from cryptoval.config import Asset, Settings, load_config
from cryptoval.contracts.checks import check_volume_anomaly, run_checks
from cryptoval.contracts.identities import check_cointime_complement, run_identity_checks
from cryptoval.contracts.loader import Contract, load_contract
from cryptoval.econometrics.registry import PAIRS
from cryptoval.econometrics.runner import run_pairs
from cryptoval.econometrics.series import LEVEL_SERIES, build_levels
from cryptoval.io.bronze import write_bronze, write_gold, write_silver
from cryptoval.manifest import RunManifest
from cryptoval.methods.registry import METHODS, run_methods
from cryptoval.quarantine import QUARANTINE_COLUMNS, quarantine_impossible
from cryptoval.scd2 import current_view, merge_scd2, reconcile_report
from cryptoval.scoring.multiplicity import correct_family_wise
from cryptoval.scoring.registry import BASELINE_MODEL_ID
from cryptoval.scoring.registry import MODELS as SCORING_MODELS
from cryptoval.scoring.runner import SCORE_COLUMNS, run_walk_forward
from cryptoval.scoring.targets import HORIZONS, build_frame
from cryptoval.sources.coinmetrics import CoinMetricsSource
from cryptoval.sources.defillama import (
    DefiLlamaHistorySource,
    DefiLlamaSource,
    DefiLlamaTvlSource,
)
from cryptoval.sources.prices import CoinGeckoSource, CoinGeckoSupplySource, EquitySource
from cryptoval.utxo.derive import derive, empty_cointime, vendor_mismatches
from cryptoval.utxo.extract import (
    assert_contiguous,
    completed_runs,
    extract,
    read_history,
    read_runs,
)
from cryptoval.volatility.runner import run_volatility
from cryptoval.volatility.settings import load_vol_config

SOURCES = {
    "coinmetrics": CoinMetricsSource(),
    "defillama": DefiLlamaSource(),
    "defillama_history": DefiLlamaHistorySource(),
    "defillama_tvl": DefiLlamaTvlSource(),
    "coingecko": CoinGeckoSource(),
    "coingecko_supply": CoinGeckoSupplySource(),
    "equities": EquitySource(),
}

SILVER_TABLE = {
    "coinmetrics": "asset_metrics",
    "defillama": "protocol_revenue",
    "defillama_history": "protocol_fee_history",
    "defillama_tvl": "protocol_tvl",
    "coingecko": "market_chart",
    "coingecko_supply": "supply_metadata",
    "equities": "equity_prices",
}

KEYS = {
    "coinmetrics": ["asset", "metric_date"],
    "defillama": ["protocol", "metric_date"],
    "defillama_history": ["protocol", "metric_date"],
    "defillama_tvl": ["protocol", "metric_date"],
    "coingecko": ["asset", "metric_date"],
    "coingecko_supply": ["gecko_id"],
    "equities": ["asset", "metric_date"],
}

# Volume-anomaly guard (spec section 9.2): which value columns to watch, per source, and
# which column identifies one series. The check assumes a single series, so it must be
# applied per entity, never to the pooled frame.
VOLUME_CHECKS = {
    "defillama": ("protocol", ["total_fees_24h", "net_revenue_24h"]),
    "defillama_history": ("protocol", ["total_fees_usd", "net_revenue_usd"]),
    "coingecko": ("asset", ["volume_usd"]),
    "coinmetrics": ("asset", ["volume_reported_spot_usd_1d"]),
}

# A contract failure blocks only for an entity in the valuation universe (spec 9.2). DefiLlama
# serves 2,991 protocols of which a handful map to an asset we value; Coin Metrics and
# CoinGecko can hold an id config no longer maps (a retired ERC-20 wrapper whose data
# predates quarantine). Neither reaches a mart or a method, so a vendor error there is real
# but not ours to block on: it warns and names the id. Equities are entirely in scope.
SCOPE_KEY = {
    "defillama": "protocol",
    "defillama_history": "protocol",
    "defillama_tvl": "protocol",
    "coinmetrics": "asset",
    "coingecko": "asset",
    "coingecko_supply": "gecko_id",
}
SCOPED_SOURCES = set(SCOPE_KEY)
_SCOPE_LABEL = {
    "defillama": "defillama_slugs",
    "defillama_history": "defillama_slugs",
    "defillama_tvl": "defillama_slugs",
    "coinmetrics": "cm_id/metric_sources",
    "coingecko": "gecko_id",
    "coingecko_supply": "gecko_id",
}
_MAX_NAMED = 8


def _universe(assets: list[Asset]) -> dict[str, set[str]]:
    """The entity ids each scoped source is allowed to block on, derived from config."""
    return {
        "defillama": {slug for a in assets for slug in a.defillama_slugs},
        "defillama_history": {slug for a in assets for slug in a.defillama_slugs},
        "defillama_tvl": {slug for a in assets for slug in a.defillama_slugs},
        "coinmetrics": {i for a in assets for i in a.cm_ids},
        "coingecko": {a.gecko_id for a in assets if a.gecko_id},
        "coingecko_supply": {a.gecko_id for a in assets if a.gecko_id},
    }


def _named(ids: list[str]) -> str:
    shown = ", ".join(ids[:_MAX_NAMED])
    return shown + (f" +{len(ids) - _MAX_NAMED} more" if len(ids) > _MAX_NAMED else "")


def _contract_and_identity_checks(frame: pd.DataFrame, contract: Contract) -> list:
    return [
        *run_checks(frame, contract, now=dt.datetime.now(dt.UTC)),
        *run_identity_checks(frame),
    ]


def _unmapped_warnings(
    table: str, unscoped: pd.DataFrame, key: str, contract: Contract
) -> list[str]:
    """One warning per failing check on out-of-universe entities, naming the entities.

    Covers the identity checks too: they are blocking for a configured id, so for an
    unmapped one they must warn rather than vanish.
    """
    failed = {
        r.name: r.detail for r in _contract_and_identity_checks(unscoped, contract) if not r.passed
    }
    if not failed:
        return []
    culprits: dict[str, list[str]] = {name: [] for name in failed}
    for entity, group in unscoped.groupby(key):
        for result in _contract_and_identity_checks(group, contract):
            if not result.passed and result.name in culprits:
                culprits[result.name].append(str(entity))
    return [
        f"{table}/{name} [unmapped: {_named(culprits[name]) or 'pooled'}]: {detail}"
        for name, detail in failed.items()
    ]


def assert_unique_symbols(assets: list[Asset]) -> None:
    """Later tasks key on symbol; a duplicate silently halves one asset's coverage."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for asset in assets:
        sym = asset.symbol.upper()
        (dupes if sym in seen else seen).add(sym)
    if dupes:
        raise ValueError(f"duplicate symbol(s) in the asset universe: {sorted(dupes)}")


def _read_silver(settings: Settings, table: str) -> pd.DataFrame:
    path = pathlib.Path(settings.resolver.table("silver", table)) / "part-0.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _append_quarantine(settings: Settings, rows: pd.DataFrame) -> None:
    """Quarantine is an append-only log, not an SCD2 table: every entry is a fact about
    one run, and row growth is itself the monitored signal (spec 9.1)."""
    existing = _read_silver(settings, "quarantine")
    combined = (
        pd.concat([existing, rows], ignore_index=True)
        if not existing.empty
        else rows.reindex(columns=QUARANTINE_COLUMNS)
    )
    write_silver(settings.resolver, "quarantine", combined)


def _value_cols(frame: pd.DataFrame, key: list[str]) -> list[str]:
    return [c for c in frame.columns if c not in key and not c.startswith("_")]


def run_ingest(
    settings: Settings,
    assets: list[Asset],
    source_names: list[str],
    start: dt.date,
    end: dt.date,
) -> dict[str, dict]:
    manifest = RunManifest(settings.resolver)
    report: dict[str, dict] = {}
    # One tz-aware instant for the whole run: merge_scd2 refuses an as_of earlier than the
    # latest stored effective_start_ts, so it must be monotonic across runs.
    as_of = dt.datetime.now(dt.UTC)
    params = {"start": start.isoformat(), "end": end.isoformat()}

    with httpx.Client(timeout=settings.request_timeout_s) as client:
        for name in source_names:
            source = SOURCES[name]
            record = manifest.start(name, params)
            try:
                result = source.fetch(start, end, assets, client)
                if result.frame.empty:
                    raise RuntimeError("source returned no rows")

                write_bronze(settings.resolver, name, end, result.frame, params, record.run_id)

                table, key = SILVER_TABLE[name], KEYS[name]
                # After bronze (the PARSED frame, not the raw payload: it is already rolled
                # to the display name and trimmed, so changing a trim rule needs a re-fetch)
                # and before the merge (so the
                # stored vintage is clean): an impossible cell is nulled and logged, not carried.
                frame, quarantined = quarantine_impossible(
                    result.frame,
                    load_contract(source.contract_path),
                    name,
                    record.run_id,
                    key,
                )
                if not quarantined.empty:
                    _append_quarantine(settings, quarantined)

                merged = merge_scd2(
                    _read_silver(settings, table),
                    frame,
                    key,
                    _value_cols(frame, key),
                    as_of,
                )
                write_silver(settings.resolver, table, merged)

                # A source that survived some failures ingested real rows, but not all of
                # them: "partial" so neither the manifest nor the exit code calls it a
                # success (fix round 1, Ruling D).
                status = "partial" if result.errors else "success"
                manifest.finish(
                    record,
                    result.rows_in,
                    len(result.frame),
                    status,
                    error="; ".join(result.errors) or None,
                )
                report[name] = {
                    "status": status,
                    "rows_in": result.rows_in,
                    "rows_out": len(result.frame),
                }
                if result.errors:
                    report[name]["errors"] = list(result.errors)
            except Exception as exc:  # recorded, not swallowed: see the manifest
                manifest.finish(record, 0, 0, "failed", error=str(exc))
                report[name] = {"status": "failed", "error": str(exc)}
    return report


def _volume_anomaly_warnings(
    name: str, table: str, df: pd.DataFrame, contract: Contract, *, unmapped: bool = False
) -> list[str]:
    """Spike warnings per series. Out-of-universe series are folded into one warning per
    watched column, naming the ids, so a long tail cannot flood the report."""
    spec = VOLUME_CHECKS.get(name)
    if spec is None or "metric_date" not in df.columns:
        return []
    entity_col, value_cols = spec
    if entity_col not in df.columns:
        return []
    flagged: list[str] = []
    for value_col in value_cols:
        if value_col not in df.columns:
            continue
        hits: list[tuple[str, str]] = []
        for entity, series in df.groupby(entity_col):
            result = check_volume_anomaly(series, contract, value_col, "metric_date")
            if not result.passed:
                hits.append((str(entity), result.detail))
        if unmapped and hits:
            flagged.append(
                f"{table}/volume_anomaly[{value_col}] "
                f"[unmapped: {_named([e for e, _ in hits])}]: {hits[0][1]}"
            )
        elif not unmapped:
            flagged.extend(
                f"{table}/volume_anomaly[{e}:{value_col}]: {detail}" for e, detail in hits
            )
    return flagged


def _stale_entity_warnings(table: str, df: pd.DataFrame, key: str) -> list[str]:
    """One entity dead while its neighbours keep reporting is invisible to a pooled check.

    Freshness contracts watch the table's own max(metric_date), so a table that is fresh
    because 18 of 19 protocols still publish says nothing about the 19th. Synthetix V3
    stopped a month before the rest and `validate` reported nothing at all.
    """
    if key not in df.columns or "metric_date" not in df.columns:
        return []
    latest = pd.to_datetime(df["metric_date"]).max()
    per_entity = pd.to_datetime(df.groupby(key)["metric_date"].max())
    lag = (latest - per_entity).dt.days
    # Two days, not one: several adapters publish a day behind, which is routine.
    stale = lag[lag > 2].sort_values(ascending=False)
    return [
        f"{table}/stale_entity[{entity}]: last row {int(days)} days behind the table's "
        f"latest ({latest.date()})"
        for entity, days in stale.items()
    ]


@dataclasses.dataclass(frozen=True)
class Validation:
    """Blocking failures and non-blocking warnings, kept apart.

    A volume anomaly warns: on live data a genuine market event (Bitcoin Cash's fork
    day, +246x against its trailing mean) is indistinguishable by ratio from the
    vendor's impossible values, and a gate that fires on healthy data is no gate
    (fix round 1, Ruling J). Impossibility is blocked by the contract's range
    maximum instead.
    """

    failures: list[str]
    warnings: list[str]


def run_validate(settings: Settings, assets: list[Asset]) -> Validation:
    failures: list[str] = []
    warnings: list[str] = []
    universe = _universe(assets)

    for name, table in SILVER_TABLE.items():
        history = _read_silver(settings, table)
        if history.empty:
            failures.append(f"{table}: no data ingested")
            continue
        # Superseded vintages are kept by design; contracts apply to what is current.
        df = current_view(history)
        contract = load_contract(SOURCES[name].contract_path)

        scoped, unscoped = df, df.iloc[0:0]
        key = SCOPE_KEY.get(name)
        if key and key in df.columns:
            in_scope = universe[name]
            if not in_scope:
                warnings.append(
                    f"no {_SCOPE_LABEL[name]} configured on any asset, so every {key} is "
                    "unmapped and its contract failures only warn"
                )
            mask = df[key].isin(in_scope)
            scoped, unscoped = df[mask], df[~mask]

        for result in run_checks(scoped, contract, now=dt.datetime.now(dt.UTC)):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")
        for result in run_identity_checks(scoped):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")

        if not unscoped.empty:
            warnings.extend(_unmapped_warnings(table, unscoped, key, contract))

        if key and key in scoped.columns:
            warnings.extend(_stale_entity_warnings(table, scoped, key))
        warnings.extend(_volume_anomaly_warnings(name, table, scoped, contract))
        if not unscoped.empty:
            warnings.extend(
                _volume_anomaly_warnings(name, table, unscoped, contract, unmapped=True)
            )
    failures.extend(_cointime_failures(settings))
    return Validation(failures, warnings)


# What _read_panel pulls from each mart beyond the asset panel. Explicit, not `select *`:
# fct_protocol_revenue_daily and fct_protocol_tvl_daily both carry protocol_count and
# composition_complete, and a `*` would silently land them as protocol_count_x/_y, where a
# method naming protocol_count in `requires` would find nothing and refuse for the wrong
# reason. Adding a column here that collides raises instead -- see _widen_panel.
PANEL_REVENUE_COLUMNS = [
    "symbol",
    "metric_date",
    "total_fees_usd",
    "net_revenue_usd",
    "revenue_regime",
    "protocol_count",
]
PANEL_TVL_COLUMNS = ["symbol", "metric_date", "tvl_usd"]
PANEL_ASSET_COLUMNS = [
    "symbol",
    "sector",
    "total_supply",
    "max_supply",
    "supply_as_of",
    "genesis_date",
]


def _widen_panel(
    panel: pd.DataFrame,
    revenue: pd.DataFrame,
    tvl: pd.DataFrame,
    assets: pd.DataFrame,
) -> pd.DataFrame:
    """The asset panel plus the protocol, TVL and dimension columns, LEFT joined.

    One place, so a method reaches a new column by naming it in ``requires`` and the method
    contract does not grow a second shape. LEFT, so an asset with no protocol revenue keeps
    its row and its Family A estimates: a missing denominator is the registry's refusal to
    make, not this function's.

    A duplicated join key RAISES rather than fanning out. A silent fan-out would multiply
    every estimate for that symbol-date, and a doubled QTM estimate looks exactly like a
    real one.
    """

    def _join(wide: pd.DataFrame, frame: pd.DataFrame, keys: list[str], name: str):
        if frame.empty:
            return wide
        duplicated = int(frame.duplicated(subset=keys).sum())
        if duplicated:
            raise ValueError(
                f"{name} has {duplicated} duplicate {tuple(keys)} row(s); joining it would "
                f"fan out the panel and double-count every estimate for those days"
            )
        # pandas would suffix a shared non-key column to _x/_y and the method asking for it
        # by name would quietly find neither.
        collisions = sorted((set(wide.columns) & set(frame.columns)) - set(keys))
        if collisions:
            raise ValueError(
                f"{name} would collide with the panel on {collisions}; rename the column in "
                f"the select list rather than letting the merge suffix it"
            )
        return wide.merge(frame, on=keys, how="left")

    wide = _join(panel.copy(), revenue, ["symbol", "metric_date"], "protocol revenue")
    wide = _join(wide, tvl, ["symbol", "metric_date"], "protocol TVL")
    return _join(wide, assets, ["symbol"], "dim_asset")


def _read_panel(settings: Settings) -> pd.DataFrame:
    """The conformed panel every method reads, widened with the Family C inputs."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:

        def maybe(table: str, columns: list[str]) -> pd.DataFrame:
            # A mart that does not exist yet yields an empty frame, and the methods
            # requiring its columns refuse with a measurement rather than crashing the run.
            # CatalogException only: a mart that exists but has RENAMED a column raises
            # BinderException, and that must crash. Swallowing it would hand every
            # dependent method an absent column and dress a schema change up as a refusal.
            try:
                return con.execute(f"select {', '.join(columns)} from {table}").fetchdf()
            except duckdb.CatalogException:
                return pd.DataFrame(columns=columns)

        return _widen_panel(
            con.execute("select * from fct_asset_metrics_daily").fetchdf(),
            maybe("fct_protocol_revenue_daily", PANEL_REVENUE_COLUMNS),
            maybe("fct_protocol_tvl_daily", PANEL_TVL_COLUMNS),
            maybe("dim_asset", PANEL_ASSET_COLUMNS),
        )
    finally:
        con.close()


def _keep_unrun_methods(settings: Settings, estimates: pd.DataFrame) -> pd.DataFrame:
    """Carry forward the methods this run did not execute.

    Gold is replaced wholesale (io.bronze.write_gold), which is right for a full run and
    silently destructive for `estimate --methods <subset>`: it would leave the mart holding
    5 methods of 28, with every dbt test still green because each one passes on a table that
    is merely incomplete. The rows carried forward keep their own panel_vintage_ts, so a
    stale method is visible as a stale vintage rather than hidden behind this run's.
    """
    path = pathlib.Path(settings.data_root) / "gold" / "valuation_estimate"
    if not path.exists():
        return estimates
    stored = pd.read_parquet(path)
    retained = stored[~stored["method_id"].isin(estimates["method_id"].unique())]
    if retained.empty:
        return estimates
    return pd.concat([retained, estimates], ignore_index=True)


def run_estimate(
    settings: Settings, assets: list[Asset], method_ids: list[str] | None = None
) -> dict[str, dict]:
    panel = _read_panel(settings)
    estimates = run_methods(panel, METHODS, method_ids)
    write_gold(settings.resolver, "valuation_estimate", _keep_unrun_methods(settings, estimates))
    report: dict[str, dict] = {}
    for method_id, group in estimates.groupby("method_id"):
        report[str(method_id)] = {
            "rows": int(len(group)),
            "refusals": int((~group["applicable"].astype(bool)).sum()),
            # Separated from refusals: an exception is a defect, not a measurement, and
            # counting it as a refusal is how a crash hides inside a plausible number.
            "raised": int(group["reason"].fillna("").str.startswith("method raised").sum()),
            "symbols": int(group["symbol"].nunique()),
        }
    return report


def _read_score_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The price panel and the estimates the signals are built from.

    Separate from _read_panel because scoring needs the ESTIMATES, which the method layer
    produces and therefore never reads. Patched wholesale in tests.
    """
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        # panel_vintage_ts stamps every score row, as fct_valuation_estimate stamps its own.
        panel = con.execute(
            "select symbol, metric_date, price_usd, panel_vintage_ts from fct_asset_metrics_daily"
        ).fetchdf()
        estimates = con.execute(
            "select symbol, metric_date, method_id, fair_value, applicable "
            "from fct_valuation_estimate"
        ).fetchdf()
        return panel, estimates
    finally:
        con.close()


def _parse_horizons(text: str) -> list[int]:
    """Day horizons from a comma-separated string, positive integers, de-duplicated in order.

    Validated before any work because a bad horizon does not fail, it misleads: -5 makes the
    target a BACKWARD return the lagged features can partly see, and the run completes with
    significant-looking rows in gold; 0 divides by zero with no measurement; -1 collides with
    the horizon = -1 sentinel of the hypothesis-count row; a repeat duplicates every key.
    """
    horizons: list[int] = []
    for raw in text.split(","):
        try:
            horizon = int(raw)
        except ValueError:
            horizon = 0
        if horizon < 1:
            raise ValueError(
                f"invalid --horizons value {raw!r}: each horizon must be a positive integer "
                f"number of days (got {text!r})"
            )
        if horizon not in horizons:
            horizons.append(horizon)
    return horizons


def _keep_unrun_scores(
    settings: Settings, scores: pd.DataFrame, model_ids: list[str], horizons: list[int]
) -> pd.DataFrame:
    """Carry forward the (model, horizon) scores this run did not execute.

    Gold is replaced wholesale (io.bronze.write_gold), which is right for a full run and
    silently destructive for `score --models <subset>` or `--horizons <subset>`: the mart
    would hold one model of eleven with every dbt test still green, because each passes on a
    table that is merely incomplete. Same defect, same cure as _keep_unrun_methods.

    Two differences from it. The stored Holm and hypothesis-count rows are DROPPED rather than
    carried: they are functions of the whole family, so they are recomputed over the merged
    raw p-values by correct_family_wise (which refuses input that already has them). And a
    model_id = '*' row (a symbol too short to fold) is per horizon, not per model, so this
    run's horizons replace it. Carried rows keep their own panel_vintage_ts.
    """
    path = pathlib.Path(settings.data_root) / "gold" / "model_score"
    if not path.exists():
        return scores
    stored = pd.read_parquet(path)
    derived = stored["metric"].isin(["dm_p_value_holm", "hypotheses_tested"])
    reran = stored["horizon"].isin(horizons) & (
        stored["model_id"].isin(model_ids) | (stored["model_id"] == "*")
    )
    retained = stored[~derived & ~reran]
    if retained.empty:
        return scores
    return pd.concat([retained, scores], ignore_index=True)[SCORE_COLUMNS]


def run_score(
    settings: Settings,
    assets: list[Asset],
    model_ids: list[str] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, dict]:
    panel, estimates = _read_score_inputs(settings)
    models = {m: SCORING_MODELS[m] for m in model_ids} if model_ids else dict(SCORING_MODELS)
    vintage = panel["panel_vintage_ts"].max() if "panel_vintage_ts" in panel.columns else None

    run_horizons = horizons or list(HORIZONS)
    frames = []
    for horizon in run_horizons:
        frame = build_frame(panel, estimates, horizon)
        frames.append(run_walk_forward(frame, models, horizon, panel_vintage_ts=vintage))

    ran = pd.concat(frames, ignore_index=True)[SCORE_COLUMNS]
    scores = correct_family_wise(_keep_unrun_scores(settings, ran, list(models), run_horizons))
    write_gold(settings.resolver, "model_score", scores)

    report: dict[str, dict] = {}
    for model_id, group in scores[scores["model_id"] != "*"].groupby("model_id"):
        # Direction matters: a DM statistic > 0 means the model's loss was HIGHER than
        # persistence's, so a significant positive value is persistence winning.
        holm = group[(group["metric"] == "dm_p_value_holm") & (group["value"] < 0.05)]
        stat = group[group["metric"] == "dm_statistic"].set_index(["symbol", "horizon"])["value"]
        signs = stat.reindex(pd.MultiIndex.from_frame(holm[["symbol", "horizon"]])).to_numpy()
        report[str(model_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "symbols": int(group["symbol"].nunique()),
            "beat_baseline_after_holm": int((signs < 0).sum()),
            "lost_to_baseline_after_holm": int((signs > 0).sum()),
        }
    return report


def _read_econ_inputs(settings: Settings) -> pd.DataFrame:
    """The levels panel. Separate from _read_score_inputs: this chapter needs the
    FUNDAMENTALS in levels and no estimates at all. Patched wholesale in tests."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        columns = ", ".join(["symbol", "metric_date", "panel_vintage_ts", *LEVEL_SERIES.values()])
        return con.execute(f"select {columns} from fct_asset_metrics_daily").fetchdf()
    finally:
        con.close()


def run_econ(
    settings: Settings, assets: list[Asset], pair_ids: list[str] | None = None
) -> dict[str, dict]:
    """The anchoring chain over every pair and symbol, full sample, written to gold."""
    panel = _read_econ_inputs(settings)
    specs = {p: PAIRS[p] for p in pair_ids} if pair_ids else dict(PAIRS)
    vintage = panel["panel_vintage_ts"].max() if "panel_vintage_ts" in panel.columns else None

    # No _keep_unrun equivalent: unlike score, a --pairs subset REPLACES the table. The
    # pairs are three, a full run takes seconds, and a partial table nobody can detect is
    # worse than one that is obviously partial.
    table = run_pairs(build_levels(panel), specs, panel_vintage_ts=vintage)
    write_gold(settings.resolver, "econometric_test", table)

    report: dict[str, dict] = {}
    for pair_id, group in table[table["pair_id"] != "*"].groupby("pair_id"):
        holm = group[(group["metric"] == "cointegration_p_value_holm") & (group["value"] < 0.05)]
        report[str(pair_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "symbols": int(group["symbol"].nunique()),
            "cointegrated_after_holm": int(len(holm)),
        }
    return report


def _read_vol_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crypto prices and traditional prices, each (symbol, metric_date, price_usd).
    Separate frames: a shared date spine would manufacture weekend nulls (spec 8.1)."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        crypto = con.execute(
            "select symbol, metric_date, price_usd, panel_vintage_ts from fct_asset_metrics_daily"
        ).fetchdf()
        traditional = con.execute(
            "select symbol, metric_date, price_usd from fct_equity_prices_daily"
        ).fetchdf()
        return crypto, traditional
    finally:
        con.close()


def run_vol(settings: Settings, assets: list[Asset]) -> dict[str, dict]:
    """The volatility chain over every asset and pair, full sample, written to gold."""
    crypto, traditional = _read_vol_inputs(settings)
    stablecoins = frozenset(a.symbol for a in assets if a.is_stablecoin)
    vintage = crypto["panel_vintage_ts"].max() if "panel_vintage_ts" in crypto.columns else None
    table = run_volatility(crypto, traditional, load_vol_config(), stablecoins, vintage)
    write_gold(settings.resolver, "volatility_test", table)

    report: dict[str, dict] = {}
    for model_id, group in table[table["model_id"] != "*"].groupby("model_id"):
        holm = group[(group["metric"] == "p_value_holm") & (group["value"] < 0.05)]
        report[str(model_id)] = {
            "rows": int(len(group)),
            "refusals": int((group["metric"] == "refused").sum()),
            "symbols": int(group["symbol"].nunique()),
            "significant_after_holm": int(len(holm)),
        }
    return report


def _btc_history(resolver) -> pd.DataFrame:
    """BTC's full Coin Metrics history as (metric_date, price_usd, supply, realized_cap)."""
    raw = read_history(resolver)
    if raw.empty:
        return pd.DataFrame(columns=["metric_date", "price_usd", "supply", "realized_cap"])
    btc = raw[raw["asset"] == "btc"]
    return pd.DataFrame(
        {
            "metric_date": pd.to_datetime(btc["metric_date"], utc=True).dt.date,
            "price_usd": pd.to_numeric(btc["PriceUSD"], errors="coerce"),
            "supply": pd.to_numeric(btc["SplyCur"], errors="coerce"),
            "realized_cap": pd.to_numeric(btc["realized_cap"], errors="coerce"),
        }
    )


def run_utxo_extract(settings: Settings, assets: list[Asset], dry_run: bool) -> dict:
    """One staged BigQuery extraction (spec 6.3). Spends budget; never part of `make all`."""
    from cryptoval.utxo.bigquery import BigQueryClient

    resolver = settings.resolver
    manifest = RunManifest(resolver)
    btc = [a for a in assets if a.symbol == "BTC"]

    def history(start: dt.date, end: dt.date) -> pd.DataFrame:
        source = CoinMetricsSource(max_retries=settings.max_retries)
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            return source.fetch(start, end, btc, client).frame

    now = dt.datetime.now(dt.UTC)
    return extract(
        resolver,
        BigQueryClient(settings.bigquery_project),
        ByteBudgetGate(manifest, settings.bigquery_monthly_byte_budget),
        manifest,
        now.date(),
        now,
        history,
        dry_run=dry_run,
    )


def run_utxo_derive(settings: Settings, assets: list[Asset]) -> dict:
    """Silver btc_cointime_daily from every complete run; empty until the backfill is whole."""
    resolver = settings.resolver
    runs = completed_runs(resolver)
    if not runs:
        write_silver(resolver, "btc_cointime_daily", empty_cointime())
        return {"status": "awaiting_backfill"}
    end = assert_contiguous(runs)
    frames = read_runs(resolver, runs)
    prices = _btc_history(resolver).set_index("metric_date")["price_usd"]
    table = derive(
        frames["btc_utxo_spent"], frames["btc_utxo_created"], frames["btc_blocks"], prices, end
    )
    write_silver(resolver, "btc_cointime_daily", table)
    last = table.iloc[-1]
    return {
        "status": "derived",
        "rows": len(table),
        "through": str(end),
        "liveliness": round(float(last["liveliness"]), 4),
        "utxo_supply": round(float(last["utxo_supply"]), 2),
    }


def _cointime_failures(settings: Settings) -> list[str]:
    cointime = _read_silver(settings, "btc_cointime_daily")
    if cointime.empty:
        return []  # awaiting backfill: the family refuses, so there is nothing to check
    result = check_cointime_complement(cointime)
    out = [] if result.passed else [f"btc_cointime_daily/{result.name}: {result.detail}"]
    cointime = cointime.assign(metric_date=pd.to_datetime(cointime["metric_date"]).dt.date)
    return out + vendor_mismatches(cointime, _btc_history(settings.resolver))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cryptoval")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("ingest", "backfill", "reconcile"):
        p = sub.add_parser(cmd)
        p.add_argument("--sources", default=",".join(SOURCES))
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--trailing-days", type=int)
    sub.add_parser("validate")
    est = sub.add_parser("estimate")
    est.add_argument(
        "--methods", help="comma-separated method ids; default every registered method"
    )

    sc = sub.add_parser("score")
    sc.add_argument("--models", help="comma-separated model ids; default every registered model")
    sc.add_argument("--horizons", help="comma-separated day horizons; default 1,7,30,90,365")

    ec = sub.add_parser("econ")
    ec.add_argument("--pairs", help="comma-separated pair ids; default every registered pair")

    sub.add_parser("vol")

    ux = sub.add_parser("utxo")
    ux.add_argument("action", choices=["extract", "derive"])
    ux.add_argument("--dry-run", action="store_true", help="price the next range, run nothing")

    args = parser.parse_args(argv)
    settings, assets = load_config()
    try:
        assert_unique_symbols(assets)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "score":
        ids = [m.strip() for m in args.models.split(",")] if args.models else None
        unknown = [m for m in ids or [] if m not in SCORING_MODELS]
        if unknown:
            print(f"unknown model(s): {unknown}; known: {sorted(SCORING_MODELS)}", file=sys.stderr)
            return 2
        if ids and BASELINE_MODEL_ID not in ids:
            print(
                f"--models must include {BASELINE_MODEL_ID}: every Diebold-Mariano test is "
                f"defined against it, and without it this run drops its models' DM and Holm "
                f"rows and shrinks the correction family (requested: {ids})",
                file=sys.stderr,
            )
            return 2
        try:
            hs = _parse_horizons(args.horizons) if args.horizons is not None else None
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        for model_id, outcome in sorted(run_score(settings, assets, ids, hs).items()):
            print(model_id, outcome)
        # A model that failed to beat persistence is the expected result (spec 11.3), not
        # a failed run.
        return 0

    if args.command == "econ":
        ids = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
        unknown = [p for p in ids or [] if p not in PAIRS]
        if unknown:
            print(f"unknown pair(s): {unknown}; known: {sorted(PAIRS)}", file=sys.stderr)
            return 2
        for pair_id, outcome in sorted(run_econ(settings, assets, ids).items()):
            print(pair_id, outcome)
        # No cointegration found is the expected result, not a failed run.
        return 0

    if args.command == "utxo":
        if args.action == "extract":
            print(run_utxo_extract(settings, assets, args.dry_run))
        else:
            print(run_utxo_derive(settings, assets))
        return 0

    if args.command == "vol":
        for model_id, outcome in sorted(run_vol(settings, assets).items()):
            print(model_id, outcome)
        # No clustering or no time-varying correlation is a result, not a failed run.
        return 0

    if args.command == "estimate":
        ids = [m.strip() for m in args.methods.split(",")] if args.methods else None
        unknown = [m for m in ids or [] if m not in METHODS]
        if unknown:
            print(f"unknown method(s): {unknown}; known: {sorted(METHODS)}", file=sys.stderr)
            return 2
        for method_id, outcome in sorted(run_estimate(settings, assets, ids).items()):
            print(method_id, outcome)
        # A refusal carrying the measurement that caused it is a research finding, so it
        # must not read as a failed run.
        return 0

    if args.command == "validate":
        report = run_validate(settings, assets)
        for w in report.warnings:
            print(f"WARN {w}", file=sys.stderr)
        for f in report.failures:
            print(f"FAIL {f}", file=sys.stderr)
        if report.failures:
            print(f"{len(report.failures)} check(s) failed, {len(report.warnings)} warning(s)")
        elif report.warnings:
            print(f"validation passed with {len(report.warnings)} warning(s)")
        else:
            print("validation passed")
        return 1 if report.failures else 0

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"unknown source(s): {unknown}; known: {sorted(SOURCES)}", file=sys.stderr)
        return 2

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    if args.trailing_days:
        start = end - dt.timedelta(days=args.trailing_days)
    elif args.start:
        start = dt.date.fromisoformat(args.start)
    else:
        start = end - dt.timedelta(days=settings.trailing_refetch_days)
    if start > end:
        print(f"start {start} is after end {end}", file=sys.stderr)
        return 2

    if args.command == "reconcile":
        # Spec 8.2 makes reconcile the detector for revisions landing outside the trailing
        # window, so a comparison that only covered some entities must not read as a clean
        # bill of health.
        incomplete = False
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            for name in names:
                result = SOURCES[name].fetch(start, end, assets, client)
                table, key = SILVER_TABLE[name], KEYS[name]
                # Silver holds the quarantined (nulled) cell, so the fetch must be cleaned the
                # same way or every quarantined cell reads as a revision forever. The log rows
                # are discarded: reconcile is a read-only diagnostic.
                fetched, _ = quarantine_impossible(
                    result.frame,
                    load_contract(SOURCES[name].contract_path),
                    name,
                    "reconcile",
                    key,
                )
                print(
                    name,
                    reconcile_report(
                        _read_silver(settings, table),
                        fetched,
                        key,
                        _value_cols(fetched, key),
                    ),
                )
                for err in result.errors:
                    incomplete = True
                    print(f"INCOMPLETE {name}: {err}", file=sys.stderr)
        return 1 if incomplete else 0

    report = run_ingest(settings, assets, names, start, end)
    for name, outcome in report.items():
        print(name, outcome)
    return 0 if all(o["status"] == "success" for o in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
