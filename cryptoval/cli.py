"""Command-line entry point: the only module that wires the pieces together."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import pathlib
import sys

import httpx
import pandas as pd

import cryptoval.methods.benchmarks  # noqa: F401 -- import registers the family
import cryptoval.methods.cost  # noqa: F401
import cryptoval.methods.monetary  # noqa: F401
import cryptoval.methods.multiples  # noqa: F401
import cryptoval.methods.network  # noqa: F401
from cryptoval.config import Asset, Settings, load_config
from cryptoval.contracts.checks import check_volume_anomaly, run_checks
from cryptoval.contracts.identities import run_identity_checks
from cryptoval.contracts.loader import Contract, load_contract
from cryptoval.io.bronze import write_bronze, write_gold, write_silver
from cryptoval.manifest import RunManifest
from cryptoval.methods.registry import METHODS, run_methods
from cryptoval.quarantine import QUARANTINE_COLUMNS, quarantine_impossible
from cryptoval.scd2 import current_view, merge_scd2, reconcile_report
from cryptoval.sources.coinmetrics import CoinMetricsSource
from cryptoval.sources.defillama import (
    DefiLlamaHistorySource,
    DefiLlamaSource,
    DefiLlamaTvlSource,
)
from cryptoval.sources.prices import CoinGeckoSource, CoinGeckoSupplySource, EquitySource

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
    return Validation(failures, warnings)


def _read_panel(settings: Settings) -> pd.DataFrame:
    """The conformed panel, read from the mart DuckDB built."""
    import duckdb

    con = duckdb.connect(str(pathlib.Path(settings.data_root) / "cryptoval.duckdb"), read_only=True)
    try:
        return con.execute("select * from fct_asset_metrics_daily").fetchdf()
    finally:
        con.close()


def run_estimate(
    settings: Settings, assets: list[Asset], method_ids: list[str] | None = None
) -> dict[str, dict]:
    panel = _read_panel(settings)
    estimates = run_methods(panel, METHODS, method_ids)
    write_gold(settings.resolver, "valuation_estimate", estimates)
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

    args = parser.parse_args(argv)
    settings, assets = load_config()
    try:
        assert_unique_symbols(assets)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

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
