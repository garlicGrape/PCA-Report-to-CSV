"""
Layers 1-2, updated for the three-layer record.

Property layer: type/range/category checks + snippet grounding (unchanged idea).
Systems/components layers: validated by RECONCILIATION — the report states its
own totals, so the sums of what we extracted must match them:
  property immediate total == Σ systems immediate == Σ Table-1 line items
  property reserves total  == Σ systems reserves  == Σ Table-2 line items
  each reserve row: Σ year_1..12 ≈ total_cost, and rul <= eul
Reconciliation is a *stronger* check than per-cell snippets for tables: a
missed row, hallucinated row, or wrong number breaks a sum.
"""
import re
import pdfplumber

from schema import (PROPERTY_FIELDS, PROPERTY_META, SYSTEM_META,
                    COMPONENT_META, RECONCILE_REL_TOL, CONFIDENCE_FLOOR)


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = re.sub(r"[^0-9.\-]", "", x)
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _meta_issues(value, meta, field, where):
    issues = []
    if value is None:
        return issues
    t = meta.get("type")
    if t == "number":
        n = _num(value)
        if n is None:
            issues.append({"where": where, "field": field, "kind": "type",
                           "detail": f"not numeric: {value!r}"})
        else:
            if "min" in meta and n < meta["min"]:
                issues.append({"where": where, "field": field, "kind": "range",
                               "detail": f"{n} < {meta['min']}"})
            if "max" in meta and n > meta["max"]:
                issues.append({"where": where, "field": field, "kind": "range",
                               "detail": f"{n} > {meta['max']}"})
    elif t == "category":
        if str(value).strip().lower() not in meta["allowed"]:
            issues.append({"where": where, "field": field, "kind": "category",
                           "detail": f"{value!r} not in {meta['allowed']}"})
    elif t == "date":
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)):
            issues.append({"where": where, "field": field, "kind": "date",
                           "detail": f"bad date: {value!r}"})
    return issues


def deterministic_checks(record: dict) -> list[dict]:
    issues = []
    prop = record["property"]

    for field in PROPERTY_FIELDS:
        meta = PROPERTY_META.get(field, {})
        issues += _meta_issues(prop[field].get("value"), meta, field, "property")

    for i, row in enumerate(record["systems"]):
        for field, meta in SYSTEM_META.items():
            issues += _meta_issues(row.get(field), meta, field, f"systems[{i}]")

    for i, row in enumerate(record["components"]):
        for field, meta in COMPONENT_META.items():
            issues += _meta_issues(row.get(field), meta, field, f"components[{i}]")
        # physics: remaining life can't exceed expected life
        eul, rul = _num(row.get("eul_years")), _num(row.get("rul_years"))
        if eul is not None and rul is not None and rul > eul:
            issues.append({"where": f"components[{i}]", "field": "rul_years",
                           "kind": "physics",
                           "detail": f"RUL {rul} > EUL {eul}: {row.get('description')!r}"})
    return issues


def _close(a, b, rel=RECONCILE_REL_TOL):
    if a is None or b is None:
        return True   # can't compare -> not a reconciliation failure
    return abs(a - b) <= max(1.0, abs(a) * rel)


def reconciliation_checks(record: dict) -> list[dict]:
    issues = []
    prop = record["property"]
    p_imm = _num(prop["immediate_repairs_total_usd"].get("value"))
    p_res = _num(prop["reserves_total_uninflated_usd"].get("value"))

    s_imm = sum(filter(None, (_num(r.get("immediate_repairs_usd"))
                              for r in record["systems"]))) or None
    s_res = sum(filter(None, (_num(r.get("replacement_reserves_usd"))
                              for r in record["systems"]))) or None

    c_imm = sum(filter(None, (_num(r.get("total_cost_usd"))
                              for r in record["components"]
                              if r.get("table") == "immediate"))) or None
    c_res = sum(filter(None, (_num(r.get("total_cost_usd"))
                              for r in record["components"]
                              if r.get("table") == "reserve"))) or None

    for name, stated, summed in [
        ("immediate: property vs systems", p_imm, s_imm),
        ("immediate: property vs components", p_imm, c_imm),
        ("reserves: property vs systems", p_res, s_res),
        ("reserves: property vs components", p_res, c_res),
    ]:
        if not _close(stated, summed):
            issues.append({"where": "cross-table", "field": name,
                           "kind": "reconcile",
                           "detail": f"stated {stated} != summed {summed}"})

    # each reserve row: year columns should sum to its total
    for i, row in enumerate(record["components"]):
        if row.get("table") != "reserve":
            continue
        years = [_num(row.get(f"year_{y}")) for y in range(1, 13)]
        ysum = sum(v for v in years if v is not None)
        total = _num(row.get("total_cost_usd"))
        if total is not None and ysum > 0 and not _close(total, ysum):
            issues.append({"where": f"components[{i}]", "field": "total_cost_usd",
                           "kind": "reconcile",
                           "detail": f"year cols sum {ysum} != total {total}: "
                                     f"{row.get('description')!r}"})
    return issues


def _page_texts(pdf_path: str) -> list[str]:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            out.append((pg.extract_text() or "").lower())
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def grounding_checks(record: dict, pdf_path: str) -> list[dict]:
    """Property layer only: cited snippet must appear on the cited page."""
    pages = _page_texts(pdf_path)
    fails = []
    for field in PROPERTY_FIELDS:
        cell = record["property"][field]
        if cell.get("value") is None or not cell.get("snippet"):
            continue
        page = cell.get("page")
        hay = pages[page - 1] if isinstance(page, int) and 1 <= page <= len(pages) \
            else " ".join(pages)
        if _norm(cell["snippet"]) not in _norm(hay):
            fails.append({"where": "property", "field": field, "kind": "grounding",
                          "detail": f"snippet not on page {page}: {cell['snippet']!r}"})
    return fails


def collect_suspect_property_fields(record, det_issues, recon_issues, ground_fails):
    """Property fields worth an LLM-judge pass. Table problems route to
    needs_review directly — a judge can't repair a 60-row table reliably."""
    suspects = {i["field"] for i in det_issues if i["where"] == "property"}
    suspects |= {g["field"] for g in ground_fails}
    for field in PROPERTY_FIELDS:
        cell = record["property"][field]
        conf = cell.get("confidence")
        if cell.get("value") is not None and isinstance(conf, (int, float)) \
                and conf <= CONFIDENCE_FLOOR:
            suspects.add(field)
    return sorted(suspects)
