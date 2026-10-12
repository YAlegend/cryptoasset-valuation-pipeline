"""Command-line entry point: the only module that wires the pieces together."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import pathlib
import sys

import httpx
import pandas as pd

from cryptoval.config import Asset, Settings, load_config
from cryptoval.contracts.checks import check_volume_anomaly, run_checks
from cryptoval.contracts.identities import run_identity_checks
from cryptoval.contracts.loader import Contract, load_contract
from cryptoval.io.bronze import write_bronze, write_silver
from cryptoval.manifest import RunManifest
from cryptoval.quarantine import QUARANTINE_COLUMNS, quarantine_impossible
from cryptoval.scd2 import current_view, merge_scd2, reconcile_report
from cryptoval.sources.coinmetrics import CoinMetricsSource
from cryptoval.sources.defillama import DefiLlamaSource
from cryptoval.sources.prices import CoinGeckoSource, EquitySource

SOURCES = {
    "coinmetrics": CoinMetricsSource(),
    "defillama": DefiLlamaSource(),
    "coingecko": CoinGeckoSource(),
    "equities": EquitySource(),
}

SILVER_TABLE = {
    "coinmetrics": "asset_metrics",
    "defillama": "protocol_revenue",
    "coingecko": "market_chart",
    "equities": "equity_prices",
}

KEYS = {
    "coinmetrics": ["asset", "metric_date"],
    "defillama": ["protocol", "metric_date"],
    "coingecko": ["asset", "metric_date"],
    "equities": ["asset", "metric_date"],
}

# Volume-anomaly guard (spec section 9.2): which value columns to watch, per source, and
# which column identifies one series. The check assumes a single series, so it must be
# applied per entity, never to the pooled frame.
VOLUME_CHECKS = {
    "defillama": ("protocol", ["total_fees_24h", "net_revenue_24h"]),
    "coingecko": ("asset", ["volume_usd"]),
    "coinmetrics": ("asset", ["volume_reported_spot_usd_1d"]),
}

# Only DefiLlama has a long tail: it serves 2,991 protocols of which a handful map to an
# asset we value. A contract failure on an unmapped protocol is real but not ours to
# block on, so it warns (spec 9.2). Every other source is entirely in scope.
SCOPED_SOURCES = {"defillama"}
SCOPE_KEY = {"defillama": "protocol"}


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
                # After bronze (which keeps what the vendor sent) and before the merge (so the
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
    name: str, table: str, df: pd.DataFrame, contract: Contract
) -> list[str]:
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
        for entity, series in df.groupby(entity_col):
            result = check_volume_anomaly(series, contract, value_col, "metric_date")
            if not result.passed:
                flagged.append(f"{table}/volume_anomaly[{entity}:{value_col}]: {result.detail}")
    return flagged


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
    in_scope = {slug for a in assets for slug in a.defillama_slugs}
    if not in_scope:
        warnings.append(
            "no defillama_slugs configured on any asset, so every protocol is unmapped "
            "and its contract failures only warn"
        )

    for name, table in SILVER_TABLE.items():
        history = _read_silver(settings, table)
        if history.empty:
            failures.append(f"{table}: no data ingested")
            continue
        # Superseded vintages are kept by design; contracts apply to what is current.
        df = current_view(history)
        contract = load_contract(SOURCES[name].contract_path)

        scoped, unscoped = df, df.iloc[0:0]
        if name in SCOPED_SOURCES:
            key = SCOPE_KEY[name]
            if key in df.columns:
                mask = df[key].isin(in_scope)
                scoped, unscoped = df[mask], df[~mask]

        for result in run_checks(scoped, contract, now=dt.datetime.now(dt.UTC)):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")
        for result in run_identity_checks(scoped):
            if not result.passed:
                failures.append(f"{table}/{result.name}: {result.detail}")

        if not unscoped.empty:
            for result in run_checks(unscoped, contract, now=dt.datetime.now(dt.UTC)):
                if not result.passed:
                    warnings.append(f"{table}/{result.name} [unmapped]: {result.detail}")

        warnings.extend(_volume_anomaly_warnings(name, table, df, contract))
    return Validation(failures, warnings)


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

    args = parser.parse_args(argv)
    settings, assets = load_config()
    try:
        assert_unique_symbols(assets)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

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
