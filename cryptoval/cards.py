"""Method cards (spec 12, deliverable 4).

One card per method: what it computes, from which columns, under what assumptions, how it
fails, and what it cites. The circularity warning is generated rather than written, because
a warning that has to be remembered is a warning that will be missing from exactly the
card where it matters.

Failure modes and assumptions are judgment, so they are hand-written -- but they live in
``config/methods/<id>.yml`` beside the spec they describe, not in the generated card:
``make cards`` rewrites every card, and prose stored in the output is prose that one
regeneration deletes.
"""

from __future__ import annotations

import pathlib

import yaml

from cryptoval.methods.base import MethodSpec, circularity

_TEMPLATE = """# {method_id}

**Family:** {family}
**Citation:** {citation}
**Returns a fair value:** {returns_fair_value}

## Data dependencies

{requires}

Minimum coverage density: **{min_density:.0%}** per column. Minimum observations:
**{min_rows}**. Below either threshold the method returns `applicable=False` with the
measured density in its reason, rather than a number fitted on too little data.

## Parameters

{params}
{circular_section}
## Failure modes

{failure_modes}

## Assumptions

{assumptions}
"""

_CIRCULAR = """
## Circular inputs

This method consumes **{columns}**, each of which is itself computed from market
capitalisation. Its output is therefore **not independent evidence** of value: it is a
diagnostic or a cost basis, and the report must treat it as one (spec section 14,
"method circularity").
"""

_SECTIONS = ("failure_modes", "assumptions")


def load_card_extras(config_dir: str = "config/methods") -> dict[str, dict]:
    """The hand-written sections, read from the method configs."""
    extras: dict[str, dict] = {}
    for path in sorted(pathlib.Path(config_dir).glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        extras[raw["method_id"]] = {k: list(raw.get(k) or []) for k in _SECTIONS}
    return extras


def render_card(spec: MethodSpec, extras: dict) -> str:
    circular = circularity(spec)
    return _TEMPLATE.format(
        method_id=spec.method_id,
        family=spec.family,
        citation=spec.citation or "(none recorded)",
        returns_fair_value="yes" if spec.returns_fair_value else "no, it is a correction",
        requires="\n".join(f"- `{c}`" for c in spec.requires) or "- (none)",
        min_density=spec.min_density,
        min_rows=spec.min_rows,
        params="\n".join(f"- `{k}` = `{v!r}`" for k, v in sorted(spec.params.items()))
        or "- (none)",
        circular_section=_CIRCULAR.format(columns=", ".join(f"`{c}`" for c in circular))
        if circular
        else "",
        failure_modes="\n".join(f"- {f}" for f in extras.get("failure_modes", []))
        or "- (to be written)",
        assumptions="\n".join(f"- {a}" for a in extras.get("assumptions", []))
        or "- (to be written)",
    )


def write_cards(
    specs: dict[str, MethodSpec], out_dir: str, extras: dict | None = None
) -> list[str]:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extras = extras or {}
    written = []
    for method_id, spec in sorted(specs.items()):
        path = out / f"{method_id}.md"
        path.write_text(render_card(spec, extras.get(method_id, {})), encoding="utf-8")
        written.append(str(path))
    return written
