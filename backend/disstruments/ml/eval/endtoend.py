"""Score an app report against a labelled record: what the user sees vs the answer key.

Used for the golden set (docs/debriefs/golden.md). Every report entry is shown (the
backend drops < 0.30); it reads as fact only at >= 0.60, when it is not a subtype entry and
when it is not flagged `calibrated: false` (both are always "?", as in the frontend's
`isUncertain`; entries without the flag, as in v1 reports, count as calibrated). Each entry maps to the
taxonomy node in its own `node` field. Against the record:
- hit: the node is positive (labels are ancestor-closed);
- false alarm: the node is observed and negative;
- unlabelled: the node is not observed (the labeller was unsure / said "type not sure").
"""
from __future__ import annotations

from typing import Any, Iterable

from ..taxonomy import Taxonomy

CONFIDENT = 0.6


def shown_as(entry: dict, confident: float = CONFIDENT) -> str:
    if entry.get("subtype") or entry.get("calibrated") is False or entry["confidence"] < confident:
        return "uncertain"
    return "confident"


def outcome(node: str | None, positive: frozenset, observed: frozenset) -> str:
    if node is None:
        return "unmapped"
    if node in positive:
        return "hit"
    return "false_alarm" if node in observed else "unlabelled"


def family(node: str, taxonomy: Taxonomy) -> str:
    """The level-1 ancestor (or the node itself)."""
    return taxonomy.ancestors(node, include_self=True)[0]


def score_entries(instruments: list[dict], positive: frozenset, observed: frozenset,
                  taxonomy: Taxonomy, confident: float = CONFIDENT) -> list[dict[str, Any]]:
    out = []
    for e in instruments:
        n = e.get("node")
        if n is not None and n not in taxonomy:
            n = None
        out.append({"label": e["label"], "node": n, "confidence": e["confidence"],
                    "shown_as": shown_as(e, confident), "subtype": bool(e.get("subtype")),
                    "subtype_uncertain": bool(e.get("subtype_uncertain")), "stem": e.get("stem"),
                    "model": e.get("model"), "family": None if n is None else family(n, taxonomy),
                    "outcome": outcome(n, positive, observed)})
    return out


def node_shown(entries: list[dict], node: str, taxonomy: Taxonomy, confident_only: bool = False) -> bool:
    """Does any entry name `node` or one of its descendants (a subtype names its family too)?"""
    sub = set(taxonomy.descendants(node, include_self=True))
    return any(e["node"] in sub and (not confident_only or e["shown_as"] == "confident")
               for e in entries)


def truth(node: str, positive: frozenset, observed: frozenset) -> str:
    return "present" if node in positive else ("absent" if node in observed else "unknown")


def node_outcomes(entries: list[dict], nodes: Iterable[str], positive: frozenset,
                  observed: frozenset, taxonomy: Taxonomy) -> dict[str, dict[str, Any]]:
    """Per node (e.g. each level-1 family): the label truth and whether the app showed it at
    all / as confident."""
    return {n: {"truth": truth(n, positive, observed),
                "shown": node_shown(entries, n, taxonomy),
                "shown_confident": node_shown(entries, n, taxonomy, confident_only=True)}
            for n in nodes}


def subtype_agreement(entries: list[dict], positive: frozenset, observed: frozenset) -> list[dict]:
    """Named subtypes (subtype entries): agree / disagree / unlabelled vs the record."""
    out = []
    for e in entries:
        if not e["subtype"]:
            continue
        o = e["outcome"]
        out.append({**e, "agreement": {"hit": "agree", "false_alarm": "disagree"}.get(o, "unlabelled")})
    return out
