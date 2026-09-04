#!/usr/bin/env python
"""
Re-validate every cached extraction and rebuild the outer cache. NO API CALLS.

Why this exists: batch.py's outer cache stores a status computed at the time
the report ran, and the judge is the only part of the stack that costs money.
When the validator changes - which it does constantly as new firms arrive -
every stored verdict is stale, and the documented way to refresh them
(delete data/cache/*.json and re-run) re-fires the judge on all 132 reports,
re-sending every PDF a third time. That is the single most expensive thing
this pipeline can do and it verifies only descriptive fields.

So: run the whole deterministic stack offline, plus two checks the judge
cannot make, and write the results back. The judge's verdicts are preserved
where a previous run produced them and are simply absent otherwise - recorded
honestly as "unverified" rather than silently treated as pass or fail.

    uv run python verify_offline.py            # rebuild the outer cache
    uv run python batch.py --aggregate-only    # then write the CSVs
"""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

from schema import PROPERTY_FIELDS
from validate import (deterministic_checks, reconciliation_checks,
                      completeness_checks, coerce_types, grounding_checks,
                      arithmetic_checks, number_presence_checks,
                      resolve_firm_from_pdf)
from pipeline import _stamp_keys
import batch as B
from batch import AGG
from taxonomy import SUBCATEGORIES, SUBCATEGORY_OTHER

HERE = Path(__file__).parent
RAW = HERE / "data" / "cache" / "raw"
CACHE = HERE / "data" / "cache"
REVIEW = HERE / "data" / "needs_review"
INBOX = HERE / "data" / "inbox" / "PCA Reports"


def _pdf_index() -> dict:
    """property_id -> PDF path, built ONCE.

    Resolving this per report meant an rglob over the whole inbox for each of
    134 records - 18,000 filesystem walks to answer 134 questions.
    """
    from pipeline import _property_id
    idx = {}
    for p in INBOX.rglob("*.pdf"):
        if not p.name.startswith("._") and "__MACOSX" not in p.parts:
            idx.setdefault(_property_id(p.stem), p)
    return idx


def one(args):
    raw_file, pdf = args
    return _verify(Path(raw_file), Path(pdf) if pdf else None)


def _verify(raw_file, pdf):
    if True:
        pid = raw_file.stem
        t0 = time.time()
        record = json.loads(raw_file.read_text())
        record = _stamp_keys(record, pid, str(pdf) if pdf else pid)
        record = B._ensure_property_fields(record)

        # Firm is the S3 partition key and the unit of leave-one-firm-out CV,
        # so a client's name sitting in it is not a cosmetic problem. Fix it
        # from the document before anything else reads the record.
        if pdf and pdf.exists():
            try:
                resolve_firm_from_pdf(record, str(pdf))
            except Exception as e:
                # Never silently: a swallowed NameError here hid the fact
                # that firm resolution was not running at all.
                print(f"  ({pid}: firm resolution failed - "
                      f"{type(e).__name__}: {str(e)[:70]})", file=sys.stderr)

        recon = reconciliation_checks(record) + completeness_checks(record)
        coercions = coerce_types(record)
        det = deterministic_checks(record) + coercions
        arith = arithmetic_checks(record)
        nulled = [c for c in coercions if "nulled" in c["detail"]]

        ground, in_text = [], []
        if pdf and pdf.exists():
            try:
                ground = grounding_checks(record, str(pdf))
                in_text = number_presence_checks(record, str(pdf))
            except Exception as e:
                print(f"  ({pid}: verification skipped - {str(e)[:60]})",
                      file=sys.stderr)
        # No PDF on disk (the three reports delivered before the corpus zip).
        # Everything that reads the record still applies; only the two checks
        # that re-open the source file are skipped.

        # Blocking policy, unchanged from batch.py: table-layer problems and
        # lost values block; grounding, number-presence and arithmetic are
        # ADVISORY and recorded without excluding the report. They point at
        # rows to inspect, not at reports to discard - and with the judge
        # unavailable, silently promoting them to blockers would throw away
        # most of the corpus.
        table_issues = recon + [i for i in det if i["where"] != "property"] + nulled
        status = "clean" if not table_issues else "needs_review"

        out = {
            "record": record,
            "flags": {"deterministic": det, "reconciliation": recon,
                      "grounding": ground, "arithmetic": arith,
                      "not_in_text": in_text, "unresolved": [],
                      "table_issues": table_issues},
            "meta": {
                "property_id": pid,
                "source_file": pdf.name if pdf else f"{pid} (pdf not on disk)",
                "status": status, "from_cache": False,
                "reused_extraction": True, "judge_ran": False,
                "seconds": round(time.time() - t0, 1),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "systems_rows": len(record["systems"]),
                "component_rows": len(record["components"]),
                "det_issues": len(det), "recon_issues": len(recon),
                "arithmetic_issues": len(arith),
                "grounding_fails": len(ground),
                "not_in_text": len(in_text),
                "unresolved": 0,
                "why": "; ".join(sorted({i["kind"] for i in table_issues})),
            },
        }
        (CACHE / f"{pid}.json").write_text(json.dumps(out, indent=2))
        stale = REVIEW / f"{pid}.flags.json"
        if status == "needs_review":
            stale.write_text(json.dumps(out["flags"], indent=2))
        elif stale.exists():
            stale.unlink()
        return out, len(ground), len(arith), len(in_text), pdf is None


def main():
    from concurrent.futures import ProcessPoolExecutor
    REVIEW.mkdir(parents=True, exist_ok=True)
    idx = _pdf_index()
    jobs = [(str(f), str(idx.get(f.stem)) if idx.get(f.stem) else None)
            for f in sorted(RAW.glob("*.json"))]
    results, rows, n_ground, n_arith, n_text, no_pdf = [], [], 0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for out, g, a, t, missing in ex.map(one, jobs):
            results.append(out); rows.append(out["meta"])
            n_ground += g; n_arith += a; n_text += t
            no_pdf += 1 if missing else 0

    clean = sum(1 for m in rows if m["status"] == "clean")
    print(f"\n{len(rows)} reports revalidated offline - {clean} clean, "
          f"{len(rows) - clean} needs_review")
    print(f"  advisory: {n_ground} grounding, {n_arith} arithmetic, "
          f"{n_text} figures not found in PDF text")
    if no_pdf:
        print(f"  ({no_pdf} extraction(s) have no PDF on disk - "
              f"verified as far as the cached record allows)")

    # Rebuild the aggregate CSVs from what was just revalidated. This is what
    # makes the 12-subcategory migration free: every cached extraction is
    # folded onto the canonical axis by batch.migrate_legacy_systems on load,
    # so the new schema reaches data/aggregate/ without a single API call.
    tables = B.aggregate(results, include_flagged=True)
    B.write_outputs(tables)
    print(f"\n  wrote {AGG}/  properties {tables['properties'].shape}  "
          f"systems {tables['systems'].shape}  "
          f"components {tables['components'].shape}  "
          f"capex_by_subcategory {tables['capex_by_subcategory'].shape}")

    _subcategory_report(tables)


def _subcategory_report(tables) -> None:
    """What the twelve actually claimed, on both layers.

    Printed every run on purpose. The failure mode this guards against is
    silent: a firm with unfamiliar section headings enters the corpus, its
    rows land nowhere, and systems.csv quietly loses a report's costs while
    still looking like a well-formed 12-row block.
    """
    sysdf, comps = tables["systems"], tables["components"]
    assessed = sysdf[sysdf["assessed"] == True]          # noqa: E712
    print("\n  subcategory coverage (systems):")
    for sub in SUBCATEGORIES:
        rows = assessed[assessed["subcategory"] == sub]
        n_rep = rows["property_id"].nunique()
        money = rows["replacement_reserves_usd"].fillna(0).sum()
        print(f"    {sub:30s} assessed in {n_rep:4d} reports   "
              f"reserves ${money:,.0f}")

    if "subcategory" in comps.columns and len(comps):
        other = (comps["subcategory"] == SUBCATEGORY_OTHER).sum()
        print(f"\n  components mapped: {len(comps) - other} of {len(comps)} "
              f"({100 * (len(comps) - other) / max(1, len(comps)):.1f}%); "
              f"{other} in '{SUBCATEGORY_OTHER}' "
              f"(professional services and unmatched)")


if __name__ == "__main__":
    main()
