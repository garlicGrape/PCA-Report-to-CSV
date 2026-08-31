"""
Batch-process a folder of PCA reports into three aggregated tables.

    python batch.py ~/PCA_Source_PDFs            # process + aggregate
    python batch.py ~/PCA_Source_PDFs --workers 4
    python batch.py --aggregate-only             # rebuild tables from cache

Why caching matters: extraction is the expensive step (~$0.50-1.00 and ~4 min
per report). Raw JSON is cached to data/cache/<id>.json, so re-running to fix
a validator, re-aggregate, or resume after an interruption costs nothing.
Delete a cache file to force re-extraction of that report.

NOTE on --aggregate-only: it rebuilds the tables from the OUTER cache
(data/cache/*.json, the validated results). It does not re-validate. After
changing a validator the sequence is: delete the outer JSONs, then run the
pipeline normally - extraction is reused from data/cache/raw/, so re-validation
costs nothing but --aggregate-only alone would find nothing to aggregate.

Outputs (data/aggregate/):
    properties.csv / .parquet     one row per report
    systems.csv    / .parquet     ~22 rows per report
    components.csv / .parquet     ~40 rows per report
    manifest.csv                  per-report status, timings, flag counts
    s3/                           same data, partitioned by firm and state

Parquet is written alongside CSV because it preserves types and the
null-vs-zero distinction that CSV flattens - important here, since a null
cost means "no work scheduled", not "$0".
"""
import argparse
import hashlib, json, re, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from schema import PROPERTY_FIELDS, SYSTEM_FIELDS, COMPONENT_FIELDS
from extract import extract
from validate import (deterministic_checks, reconciliation_checks,
                      grounding_checks, collect_suspect_property_fields,
                      completeness_checks, coerce_types)
from judge import judge_fields
from pipeline import _stamp_keys, _property_id

HERE = Path(__file__).parent
CACHE = HERE / "data" / "cache"
AGG = HERE / "data" / "aggregate"
REVIEW = HERE / "data" / "needs_review"


def _real_pdfs(folder: Path) -> list:
    """Every *.pdf under folder, minus macOS archive litter.

    Unzipping a Mac-made archive leaves __MACOSX/._Report.pdf AppleDouble
    stubs beside the real files. They match *.pdf and are not PDFs - a few KB
    of resource-fork metadata each. Left in, every one is handed to the
    extractor at roughly $1-2 per attempt before failing, and each shows up
    in the manifest as an error that has nothing to do with extraction.
    """
    return sorted(p for p in folder.rglob("*.pdf")
                  if not p.name.startswith("._")
                  and "__MACOSX" not in p.parts)


def _name_quality(path: Path) -> tuple:
    """Sort key: the copy we would rather keep sorts first."""
    stem = path.stem
    placeholder = bool(re.search(r"\b(dup|duplicate|placeholder|copy)\b",
                                 stem, re.I))
    # A stem that is mostly a job number carries no property identity.
    letters = sum(c.isalpha() for c in stem)
    return (placeholder, -letters, stem)


def _dedupe(pdfs: list) -> tuple:
    """-> (unique pdfs, [(dropped, kept), ...]).

    The inbox holds 138 files and 132 reports. Three pairs are byte-identical
    with different names ("Regency House - PCR 2018.pdf" and "Regency House
    dup placeholder.pdf"; "23 - PCA Report 2021.pdf" and "25 - PCA Report
    2021.pdf"); three more are the same report re-exported, so the bytes
    differ but the cover page does not ("24 - PCA Report 2021.pdf" is "The
    Woodlands at Hillcrest"; "2226477 - PCA 2022.pdf" is "Lofts at the
    Highlands"). Some pairs do not even share a property name, so nothing
    upstream of the file itself catches them.

    This matters twice over. Each duplicate is a second extraction of a report
    already paid for, and - the part that does not show up on the invoice - it
    puts two identical properties into the training tables. That inflates n,
    correlates the rows, and quietly breaks leave-one-property-out CV: the
    held-out property is sitting in the training fold under another name.

    Two passes, cheapest first: the full file hash is exact and catches the
    re-namings; the first page's text catches the re-exports. The text pass
    requires a substantial page AND a matching page count, so two different
    reports sharing a boilerplate cover are not mistaken for each other.
    """
    # Which copy survives is not arbitrary. property_id, property_name and the
    # S3 partition keys are all derived from the file name, so keeping "23 -
    # PCA Report 2021.pdf" over "River Bend Assisted Living & Memory Care -
    # Rochester, MN - PCA 2021.pdf" would trade an identifiable property for a
    # number. Prefer the more descriptive name, and never prefer one that
    # announces itself as a placeholder.
    pdfs = sorted(pdfs, key=_name_quality)

    seen, unique, dropped = {}, [], []
    for p in pdfs:
        # A file that cannot be read must not take the whole batch down. This
        # runs before any report is processed, so an unreadable path here -
        # a broken symlink, a permissions problem, a half-copied download -
        # aborted a 130-report run at report zero. Skip it loudly instead;
        # process_one already records per-report errors in the manifest.
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError as e:
            print(f"(unreadable, skipping: {p.name} - {e.strerror})")
            continue
        key = ("bytes", digest)
        if key in seen:
            dropped.append((p, seen[key]))
            continue
        seen[key] = p
        unique.append(p)

    seen2, out = {}, []
    for p in unique:
        try:
            reader = PdfReader(str(p))
            text = (reader.pages[0].extract_text() or "").strip() if reader.pages else ""
            n_pages = len(reader.pages)
        except Exception:
            out.append(p)
            continue
        if len(text) < 200:
            out.append(p)              # too thin to identify; keep it
            continue
        key = (hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest(), n_pages)
        if key in seen2:
            dropped.append((p, seen2[key]))
            continue
        seen2[key] = p
        out.append(p)

    return out, dropped


def _ensure_property_fields(record: dict) -> dict:
    """Backfill property cells the schema has gained since this record was
    extracted.

    data/cache/raw/ holds extractions produced against whatever PROPERTY_FIELDS
    looked like at the time. Adding a field to the schema - num_stories_min and
    num_stories_max, say - means every reused extraction is missing it, and
    anything indexing record["property"][field] directly raises KeyError on a
    report that was fine yesterday. Since new firms will keep forcing new
    fields, this backfills instead of invalidating the expensive cache.
    """
    prop = record.setdefault("property", {})
    for f in PROPERTY_FIELDS:
        cell = prop.get(f)
        if not isinstance(cell, dict):
            prop[f] = {"value": cell, "page": None, "snippet": None,
                       "confidence": None}
    return record


# ── one report ─────────────────────────────────────────────────────────────
def process_one(pdf_path: Path, use_cache: bool = True,
                use_judge: bool = True) -> dict:
    stem = pdf_path.stem
    pid = _property_id(stem)
    cache_file = CACHE / f"{pid}.json"
    t0 = time.time()

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cached["meta"]["from_cache"] = True
        return cached

    # Two-level cache. RAW is the expensive part (~$1 and 4-15 min per
    # report); validation and judging are cheap and change often as we tune.
    # Caching them separately means a judge bug or a validator tweak costs
    # nothing to re-run - only the first extraction is ever paid for.
    raw_dir = CACHE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"{pid}.json"

    try:
        if use_cache and raw_file.exists():
            record = json.loads(raw_file.read_text())
            reused_raw = True
        else:
            record = extract(str(pdf_path))
            raw_file.write_text(json.dumps(record, indent=2))
            reused_raw = False
        record = _stamp_keys(record, stem, str(pdf_path))
        record = _ensure_property_fields(record)

        det = deterministic_checks(record)
        recon = reconciliation_checks(record) + completeness_checks(record)
        ground = grounding_checks(record, str(pdf_path))
        suspects = collect_suspect_property_fields(record, det, recon, ground)

        # The judge is the most expensive layer here: it re-sends the entire
        # PDF a third time to second-opinion DESCRIPTIVE fields, and it was
        # about a third of spend across this corpus. Reconciliation, which is
        # free, is what protects the financial data. With --no-judge the
        # suspects are simply not raised - they stay as extracted, and
        # verify_offline.py re-runs every other check at no cost.
        if not use_judge:
            suspects, verdicts = [], {}
        else:
            try:
                verdicts = judge_fields(str(pdf_path), record["property"],
                                        suspects)
            except Exception as je:
                # A second opinion, not the extraction - degrade, don't die.
                verdicts = {f: {"ok": False, "corrected_value": None,
                                "reason": f"judge failed: {str(je)[:120]}"}
                            for f in suspects}
        unresolved = []
        for f in suspects:
            v = verdicts.get(f, {})
            if v.get("ok") is True:
                continue
            if v.get("corrected_value") is not None:
                record["property"][f]["value"] = v["corrected_value"]
                record["property"][f]["confidence"] = 0.99
            else:
                unresolved.append({"field": f,
                                   "reason": v.get("reason", "flagged")})

        # After judging: make numeric columns numeric, normalise firm-specific
        # category spellings, then re-derive the deterministic issues.
        #
        # Re-running rather than patching the old list. Values have changed
        # twice by this point (judge corrections, then coercion), so the
        # pre-judge issues describe a record that no longer exists. The old
        # `[i for i in det if i["kind"] != "type"]` dropped EVERY type issue,
        # including component and system ones that coercion never touches -
        # so a non-numeric total_cost_usd on a Table 2 row was flagged and then
        # silently discarded before it could block anything.
        coercions = coerce_types(record)
        det = deterministic_checks(record) + coercions
        nulled = [c for c in coercions if "nulled" in c["detail"]]

        table_issues = recon + [i for i in det if i["where"] != "property"] + nulled
        status = "clean" if not unresolved and not table_issues else "needs_review"

        out = {
            "record": record,
            "flags": {"deterministic": det, "reconciliation": recon,
                      "grounding": ground, "unresolved": unresolved,
                      "table_issues": table_issues},
            "meta": {
                "property_id": pid, "source_file": pdf_path.name,
                "status": status, "from_cache": False,
                "reused_extraction": reused_raw,
                "seconds": round(time.time() - t0, 1),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "systems_rows": len(record["systems"]),
                "component_rows": len(record["components"]),
                "det_issues": len(det), "recon_issues": len(recon),
                "why": "; ".join(sorted({i["kind"] for i in table_issues})) or
                       ("unverified fields" if unresolved else ""),
                "grounding_fails": len(ground), "unresolved": len(unresolved),
                "judge_ran": use_judge,
            },
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(out, indent=2))

        if status == "needs_review":
            REVIEW.mkdir(parents=True, exist_ok=True)
            (REVIEW / f"{pid}.flags.json").write_text(
                json.dumps(out["flags"], indent=2))
        else:
            # A report that has been fixed should stop appearing in the review
            # folder. Stale .flags.json files from earlier runs are worse than
            # no file - they describe failures that no longer exist.
            stale = REVIEW / f"{pid}.flags.json"
            if stale.exists():
                stale.unlink()
        return out

    except Exception as e:
        return {"record": None,
                "flags": {"error": str(e), "traceback": traceback.format_exc()},
                "meta": {"property_id": pid, "source_file": pdf_path.name,
                         "status": "error", "from_cache": False,
                         "seconds": round(time.time() - t0, 1),
                         "error": str(e)[:300]}}


# ── aggregation ────────────────────────────────────────────────────────────
def aggregate(results: list[dict], include_flagged: bool = False) -> dict:
    props, systems, comps, manifest = [], [], [], []
    for r in results:
        manifest.append(r["meta"])
        rec = r.get("record")
        # By default only reconciled reports enter the training tables.
        # --include-flagged adds needs_review rows too, tagged so you can
        # filter them later: never silently mix audited and unaudited data.
        status = r["meta"].get("status")
        if not rec or status == "error":
            continue
        if status != "clean" and not include_flagged:
            continue
        # .get on both levels: a cached record predates any field the schema
        # has gained since, and a missing column should be null, not a crash.
        prow = {f: (rec["property"].get(f) or {}).get("value")
                for f in PROPERTY_FIELDS}
        prow["extraction_status"] = status
        prow["recon_issues"] = r["meta"].get("recon_issues", 0)
        props.append(prow)
        systems.extend({f: row.get(f) for f in SYSTEM_FIELDS}
                       for row in rec["systems"])
        comps.extend({f: row.get(f) for f in COMPONENT_FIELDS}
                     for row in rec["components"])

    return {
        "properties": pd.DataFrame(
            props, columns=PROPERTY_FIELDS + ["extraction_status", "recon_issues"]),
        "systems": pd.DataFrame(systems, columns=SYSTEM_FIELDS),
        "components": pd.DataFrame(comps, columns=COMPONENT_FIELDS),
        "manifest": pd.DataFrame(manifest),
    }


def _slug(x) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x or "unknown")).strip("_") or "unknown"


def write_outputs(tables: dict, s3_partitions: bool = True):
    AGG.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(AGG / f"{name}.csv", index=False)
        if name != "manifest":
            try:
                df.to_parquet(AGG / f"{name}.parquet", index=False)
            except Exception as e:
                print(f"  (parquet skipped for {name}: {e})")

    if not s3_partitions or tables["properties"].empty:
        return
    # S3-style layout: partitioning by firm and state makes leave-one-firm-out
    # and leave-one-region-out CV a prefix filter instead of an in-memory one.
    root = AGG / "s3"
    props = tables["properties"]
    keys = props[["property_id", "report_firm", "state"]].copy()
    for name in ["properties", "systems", "components"]:
        df = tables[name]
        if df.empty:
            continue
        if name != "properties":
            df = df.merge(keys[["property_id", "state"]], on="property_id",
                          how="left", suffixes=("", "_p"))
        for (firm, state), grp in df.groupby(
                [df.get("report_firm"), df.get("state")], dropna=False):
            d = root / name / f"firm={_slug(firm)}" / f"state={_slug(state)}"
            d.mkdir(parents=True, exist_ok=True)
            part = grp.drop(columns=[c for c in grp.columns
                                     if c.endswith("_p")], errors="ignore")
            # Never let a serialisation problem destroy a finished run. The
            # extraction is the expensive part and it is already done by the
            # time this executes; a partition that will not encode should cost
            # that partition, not every output file. (A mixed-type column got
            # here once and took the whole 132-report run's outputs with it.)
            try:
                part.to_parquet(d / "part.parquet", index=False)
            except Exception as e:
                part.to_csv(d / "part.csv", index=False)
                print(f"  (parquet failed for {name} {firm}/{state}, wrote "
                      f"CSV instead: {str(e)[:90]})")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(HERE / "data" / "inbox"))
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel reports; keep low to respect API rate limits")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--include-flagged", action="store_true",
                    help="also aggregate needs_review reports (tagged via "
                         "extraction_status) instead of excluding them")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM judge (about a third of API spend). "
                         "Suspect property fields are recorded as unverified "
                         "instead of second-opinioned; reconciliation and the "
                         "rest of the stack are unaffected. Re-run "
                         "verify_offline.py afterwards for the free checks.")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="process byte- or cover-identical reports separately "
                         "instead of skipping them (they inflate n and break "
                         "leave-one-property-out CV)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild tables from cached JSON, no API calls")
    args = ap.parse_args()

    if args.aggregate_only:
        results = [json.loads(f.read_text()) for f in sorted(CACHE.glob("*.json"))]
        print(f"Loaded {len(results)} cached results")
        if not results:
            print("\nNothing to aggregate: data/cache/ holds no validated "
                  "results.\n--aggregate-only rebuilds the tables from that "
                  "cache; it does not re-validate.\nIf you just deleted the "
                  "outer cache to pick up a validator change, run the pipeline "
                  "normally instead:\n\n    python batch.py <folder> "
                  "--workers 3\n\nExtraction is reused from data/cache/raw/, "
                  "so that costs no API calls.")
            return
    else:
        folder = Path(args.folder).expanduser()
        pdfs = _real_pdfs(folder)
        if not pdfs:
            print(f"No PDFs found in {folder}")
            if list(folder.rglob("*.zip")):
                print("There are zip files there - unpack them first.")
            return
        skipped = len([p for p in folder.rglob("*.pdf")]) - len(pdfs)
        if skipped:
            print(f"(ignoring {skipped} macOS resource-fork stub(s))")

        if not args.keep_duplicates:
            pdfs, dupes = _dedupe(pdfs)
            for dropped, kept in dupes:
                print(f"(duplicate: {dropped.name!r} is the same report as "
                      f"{kept.name!r} - skipping)")
            if dupes:
                print(f"(skipped {len(dupes)} duplicate(s); "
                      f"{len(pdfs)} unique reports remain)")

        print(f"Processing {len(pdfs)} reports with {args.workers} workers...\n")
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, p, not args.no_cache,
                              not args.no_judge): p for p in pdfs}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                m = r["meta"]
                tag = "cached" if m.get("from_cache") else f"{m['seconds']}s"
                extra = ""
                if m["status"] == "error":
                    extra = f"  ERROR: {m.get('error','')[:80]}"
                elif m["status"] == "needs_review":
                    extra = (f"  recon={m.get('recon_issues')} "
                             f"det={m.get('det_issues')} "
                             f"unresolved={m.get('unresolved')} "
                             f"-> {m.get('why', '')}")
                print(f"[{i}/{len(pdfs)}] {m['status']:13} {tag:>8}  "
                      f"{m['source_file'][:46]}{extra}")
                results.append(r)

    tables = aggregate(results, include_flagged=args.include_flagged)
    write_outputs(tables)

    man = tables["manifest"]
    counts = (man["status"].value_counts().to_dict()
              if not man.empty and "status" in man.columns else {})
    print(f"\n{'='*62}")
    print(f"reports:     {len(man)}   {counts}")
    print(f"properties:  {len(tables['properties'])} rows")
    print(f"systems:     {len(tables['systems'])} rows")
    print(f"components:  {len(tables['components'])} rows")
    if not tables["properties"].empty:
        firms = tables["properties"]["report_firm"].value_counts().to_dict()
        print(f"firms:       {firms}")
    print(f"\nwritten to {AGG}")
    if counts.get("needs_review") or counts.get("error"):
        n = counts.get("needs_review", 0) + counts.get("error", 0)
        print(f"\n{n} report(s) excluded from the tables. Review their "
              f"flags in data/needs_review/ before treating the aggregate as "
              f"complete - silently dropping failures biases the training set.")


if __name__ == "__main__":
    main()