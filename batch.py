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
import hashlib, json, re, shutil, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from schema import (PROPERTY_FIELDS, SYSTEM_FIELDS, COMPONENT_FIELDS,
                    table_horizon)
from extract import extract, _normalise_systems, PROMPT_VERSION
from taxonomy import classify_subcategory, SUBCATEGORY_OTHER, SUBCATEGORIES
from validate import (deterministic_checks, reconciliation_checks,
                      grounding_checks, collect_suspect_property_fields,
                      collect_suspect_system_rows,
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


def migrate_legacy_systems(record: dict) -> bool:
    """Fold a pre-subcategory `systems` block onto the canonical 12 rows.

    -> True when a migration happened.

    THE POINT OF THIS FUNCTION IS THAT NOTHING HAS TO BE RE-EXTRACTED. The 134
    cached extractions in data/cache/raw/ cost real money and several hours;
    they carry one row per firm-specific section ("3.4 Roofing", "B.4.1A",
    "Walkways, Grade-Level Steps and Ramps"). Every one of those rows still
    maps deterministically onto the twelve, so the whole corpus can adopt the
    new schema for free and only FUTURE runs pay the cheaper extraction.

    The fold is lossy in exactly one respect and it is recorded rather than
    hidden: the firm's own section number and heading move into
    `source_sections`, so the row can still be traced back to the page even
    though `system_name` no longer exists as a column. Rows whose heading maps
    to no subcategory - "Utilities", "Regulatory Compliance" - are dropped
    from the systems block and counted in the return of `rollup.py`, never
    forced into one of the twelve.
    """
    rows = record.get("systems") or []
    if rows and all(r.get("subcategory") for r in rows):
        return False        # already migrated, or freshly extracted

    folded = []
    for row in rows:
        name = " ".join(str(row.get(k) or "").strip()
                        for k in ("system_name", "section_code")).strip()
        # Orphans are passed through, not dropped: _normalise_systems
        # gathers any that carry cost into a single "other" row so the systems
        # sum still ties out against the report's stated totals.
        sub = classify_subcategory(row.get("system_name") or name)
        src = " ".join(str(row.get(k) or "").strip()
                       for k in ("section_code", "system_name")).strip()
        folded.append({
            # Join keys are carried across. _stamp_keys runs BEFORE this
            # migration, so rebuilding the rows without them left every
            # systems row with a null property_id and silently unjoinable.
            "property_id": row.get("property_id"),
            "report_firm": row.get("report_firm"),
            "subcategory": sub,
            "condition": row.get("condition"),
            "condition_secondary": row.get("condition_secondary"),
            "condition_rating_numeric": row.get("condition_rating_numeric"),
            "action_required": row.get("action_required"),
            "rul_years": None,          # the legacy layer never carried one
            "immediate_repairs_usd": row.get("immediate_repairs_usd"),
            "short_term_repairs_usd": row.get("short_term_repairs_usd"),
            "non_critical_repairs_usd": row.get("non_critical_repairs_usd"),
            "replacement_reserves_usd": row.get("replacement_reserves_usd"),
            "source_sections": src or None,
            "notes": None,
            "assessed": True,
        })
    record["systems"] = _normalise_systems(folded, record.get("_source_file",
                                                             "cached record"))
    return True


def _backfill_join_keys(record: dict, property_id: str | None = None) -> int:
    """Ensure every systems/components row carries property_id and report_firm.

    -> number of rows repaired.

    `_stamp_keys` runs BEFORE the legacy systems migration, so the filler rows
    that `_normalise_systems` creates for subcategories a report never covered
    are born after the stamping and keep property_id=None. They then survive
    all the way into systems.csv, where `groupby("property_id")` silently drops
    them - three rows vanished from the corpus that way, and the only reason it
    was noticed is that 132x12 + 3x11 did not equal the 1620 rows in the file.

    A null join key is the worst kind of defect in a table whose entire purpose
    is being joined, so this runs unconditionally rather than only on the
    migration path.
    """
    fixed = 0
    prop = record.get("property") or {}
    cell = prop.get("report_firm") or {}
    firm = cell.get("value") if isinstance(cell, dict) else cell
    for layer in ("systems", "components"):
        rows = record.get(layer) or []
        pid = property_id or next(
            (r.get("property_id") for r in rows if r.get("property_id")), None)
        for r in rows:
            if not r.get("property_id") and pid:
                r["property_id"] = pid
                fixed += 1
            if not r.get("report_firm") and firm:
                r["report_firm"] = firm
    return fixed


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
    migrate_legacy_systems(record)
    _backfill_join_keys(record)
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
        # The outer cache must honour the prompt stamp too, or it short
        # circuits the check below and a re-extraction run quietly returns
        # last week's validated results for every report.
        if cached.get("meta", {}).get("prompt_version") == PROMPT_VERSION:
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
        # Reuse a cached extraction ONLY if the current prompt produced it.
        # A stale stamp means the record predates a prompt or schema change,
        # and mixing those is precisely the firm-correlated bias documented in
        # the handoff. This is also what makes a full re-extraction resumable:
        # reports already done under this prompt are skipped for free, so a
        # run killed by a usage limit is restarted with the same command
        # instead of re-paying for everything it had finished.
        record, reused_raw = None, False
        if use_cache and raw_file.exists():
            cached_raw = json.loads(raw_file.read_text())
            stamp = cached_raw.get("_prompt_version")
            if stamp == PROMPT_VERSION:
                record, reused_raw = cached_raw, True
            else:
                print(f"  ({pid}: cached extraction is from prompt "
                      f"{stamp or 'unversioned'}, current is {PROMPT_VERSION}"
                      f" - re-extracting)", file=sys.stderr)

        if record is None:
            record = extract(str(pdf_path))
            raw_file.write_text(json.dumps(record, indent=2))
        record = _stamp_keys(record, stem, str(pdf_path))
        record = _ensure_property_fields(record)

        det = deterministic_checks(record)
        recon = reconciliation_checks(record) + completeness_checks(record)
        ground = grounding_checks(record, str(pdf_path))
        suspects = collect_suspect_property_fields(record, det, recon, ground)
        sys_suspects = collect_suspect_system_rows(record, det)

        # The judge is the most expensive layer here: it re-sends the entire
        # PDF a third time to second-opinion DESCRIPTIVE fields, and it was
        # about a third of spend across this corpus. Reconciliation, which is
        # free, is what protects the financial data. With --no-judge the
        # suspects are simply not raised - they stay as extracted, and
        # verify_offline.py re-runs every other check at no cost.
        if not use_judge:
            suspects, sys_suspects, verdicts = [], [], {}
        else:
            try:
                verdicts = judge_fields(str(pdf_path), record["property"],
                                        suspects,
                                        systems=record["systems"],
                                        subcategories=tuple(sys_suspects))
            except Exception as je:
                # A second opinion, not the extraction - degrade, don't die.
                verdicts = {f: {"ok": False, "corrected_value": None,
                                "reason": f"judge failed: {str(je)[:120]}"}
                            for f in list(suspects) +
                                     [f"sys:{c}" for c in sys_suspects]}
        # Systems verdicts. A correction here is a partial ROW, not a
        # scalar, so it is merged key-by-key rather than assigned: the judge
        # is asked only about the fields it was shown, and overwriting the
        # whole row with its reply would blank everything it stayed silent on.
        unresolved = []
        for c in sys_suspects:
            v = verdicts.get(f"sys:{c}", {})
            if v.get("ok") is True:
                continue
            fix = v.get("corrected_value")
            row = next((r for r in record["systems"]
                        if r.get("subcategory") == c), None)
            if isinstance(fix, dict) and row is not None:
                for k, val in fix.items():
                    if k in SYSTEM_FIELDS and k != "subcategory":
                        row[k] = val
            else:
                unresolved.append({"field": f"systems.{c}",
                                   "reason": v.get("reason", "flagged")})

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
        # RECONCILIATION MUST BE RECOMPUTED TOO, for exactly the reason given
        # above for `det`: the judge has since rewritten values and coercion
        # has since retyped them, so the pre-judge `recon` describes a record
        # that no longer exists. Only `det` was re-run, which meant a judge
        # correction to a cost could silently break the tie-out and the report
        # would still be reported clean.
        #
        # Measured on ARIUM Apartments: the judge raised building_envelope's
        # reserves from 420,942 to 585,942, putting the systems sum 165,000
        # over the report's own stated total - and the run called it clean,
        # because reconciliation had been computed before the judge ran. Six
        # reports in this corpus were clean for this reason alone.
        #
        # A judge correction that breaks reconciliation is precisely what a
        # human should look at, so this is the check earning its keep.
        recon = reconciliation_checks(record) + completeness_checks(record)
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
                "subcategories_assessed": sum(
                    1 for r in record["systems"] if r.get("assessed")),
                "judged_subcategories": list(sys_suspects),
                "component_rows": len(record["components"]),
                "det_issues": len(det), "recon_issues": len(recon),
                "why": "; ".join(sorted({i["kind"] for i in table_issues})) or
                       ("unverified fields" if unresolved else ""),
                "grounding_fails": len(ground), "unresolved": len(unresolved),
                "judge_ran": use_judge,
                "prompt_version": record.get("_prompt_version", "unversioned"),
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
def _target_quality(flags: dict) -> str:
    """How much the capex TARGET can be trusted, per report.

    `extraction_status` is the wrong filter for capex modelling and measurably
    so: it aggregates every check in the stack, most of which do not touch the
    components-derived target at all. On this corpus 34 of the 61
    needs_review reports carry only systems-layer or property-layer flags -
    their capex target is untouched - so filtering on `clean` would discard a
    quarter of the usable training set for problems that are not in the label.

    This looks only at issues that bear on the target:
      "no_line_items" - the report states a total but no line items were
          extracted behind it, so total_capex_usd is a FALSE ZERO. This is the
          dangerous one: a $5M property labelled $0 is a poisoned label, worse
          than a missing row, and it must not be trained on.
      "does_not_tie" - line items exist but do not sum to the report's own
          stated total. The target is approximately right and biased low or
          high by a knowable amount; usable with care.
      "ok"           - nothing in the stack disputes the target.
    """
    issues = flags.get("table_issues") or []
    empty = any(i.get("where") == "components" and i.get("kind") == "empty"
                for i in issues)
    untied = any(i.get("kind") == "reconcile"
                 and "components" in str(i.get("field", ""))
                 for i in issues)
    if empty:
        return "no_line_items"
    return "does_not_tie" if untied else "ok"


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
        # Provenance on every row, not just the manifest. The training set can
        # then be filtered by which prompt produced a report - the confound the
        # handoff calls the worst problem in this corpus. "unversioned" means
        # the record predates the stamp and its prompt is unknown; it is NOT a
        # claim that it matches anything.
        prow["prompt_version"] = r["meta"].get("prompt_version") or "unversioned"
        prow["target_quality"] = _target_quality(r.get("flags") or {})
        props.append(prow)
        systems.extend({f: row.get(f) for f in SYSTEM_FIELDS}
                       for row in rec["systems"])
        comps.extend({f: row.get(f) for f in COMPONENT_FIELDS}
                     for row in rec["components"])

    tables = {
        "properties": pd.DataFrame(
            props, columns=PROPERTY_FIELDS + ["extraction_status",
                                              "recon_issues", "prompt_version",
                                              "target_quality"]),
        "systems": pd.DataFrame(systems, columns=SYSTEM_FIELDS),
        "components": pd.DataFrame(comps, columns=COMPONENT_FIELDS),
        "manifest": pd.DataFrame(manifest),
    }
    tables["capex_by_subcategory"] = _capex_by_subcategory(tables)
    return tables


def _capex_by_subcategory(tables) -> pd.DataFrame:
    """The modelling table: one row per (property, subcategory) - 12 per report.

    WHY THIS IS A SEPARATE TABLE AND NOT JUST systems.csv. The systems layer's
    money columns come from each report's executive-summary table, and most
    firms do not cost their work there - they cost it in the line-item tables.
    Measured on this corpus: `replacement_reserves_usd` is populated on 791 of
    1,633 systems rows, while `total_cost_usd` is populated on 2,408 of 2,408
    component rows. Training a capex model on the systems columns would be
    training on the half of the corpus whose firms happen to publish a costed
    summary - and firm is the CV axis, so that bias is the worst kind here.

    So the capex TARGET is summed from components, which is where the money
    actually is, and the condition FEATURES come from the systems layer, which
    is where the ratings actually are. The subcategory axis is what lets the
    two be joined at all; before it there was no shared key.

    Every property gets all 12 subcategories whether or not it has line items
    in them, because "this property has no roofing capex" is a real zero and
    dropping the row would turn it into missing data. `assessed` and
    `n_line_items` are what tell a true zero from an unextracted one.
    """
    comps, systems = tables["components"], tables["systems"]
    props = tables["properties"]
    if props.empty:
        return pd.DataFrame(columns=[
            "property_id", "report_firm", "subcategory", "reserve_capex_usd",
            "near_term_capex_usd", "n_reserve_items", "n_near_term_items",
            "condition", "condition_rating_numeric", "rul_years",
            "assessed", "extraction_status", "prompt_version",
            "target_quality"])

    # The full (property x 12) grid. Reindexing onto it is what guarantees the
    # fixed width - a groupby alone silently yields ragged output.
    grid = pd.MultiIndex.from_product(
        [props["property_id"].dropna().unique(), SUBCATEGORIES],
        names=["property_id", "subcategory"]).to_frame(index=False)

    def _roll(mask, money_col, count_col):
        sub = comps[mask]
        if sub.empty:
            return pd.DataFrame(columns=["property_id", "subcategory",
                                         money_col, count_col])
        g = (sub.groupby(["property_id", "subcategory"])["total_cost_usd"]
             .agg(["sum", "count"]).reset_index())
        return g.rename(columns={"sum": money_col, "count": count_col})

    horizon = comps["table"].map(table_horizon) if "table" in comps else None
    out = grid
    if horizon is not None:
        out = out.merge(_roll(horizon == "reserve",
                              "reserve_capex_usd", "n_reserve_items"),
                        on=["property_id", "subcategory"], how="left")
        out = out.merge(_roll(horizon == "near_term",
                              "near_term_capex_usd", "n_near_term_items"),
                        on=["property_id", "subcategory"], how="left")

    for c, fill in (("reserve_capex_usd", 0.0), ("near_term_capex_usd", 0.0),
                    ("n_reserve_items", 0), ("n_near_term_items", 0)):
        if c not in out:
            out[c] = fill
        out[c] = out[c].fillna(fill)

    feats = systems[["property_id", "subcategory", "condition",
                     "condition_rating_numeric", "rul_years", "assessed"]]
    out = out.merge(feats, on=["property_id", "subcategory"], how="left")
    out = out.merge(
        props[["property_id", "report_firm", "extraction_status",
               "prompt_version", "target_quality"]],
        on="property_id", how="left")
    out["total_capex_usd"] = (out["reserve_capex_usd"]
                              + out["near_term_capex_usd"])
    return out[["property_id", "report_firm", "subcategory",
                "reserve_capex_usd", "near_term_capex_usd", "total_capex_usd",
                "n_reserve_items", "n_near_term_items", "condition",
                "condition_rating_numeric", "rul_years", "assessed",
                "extraction_status", "prompt_version", "target_quality"]]


def _slug(x) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x or "unknown")).strip("_") or "unknown"


def _parquet_safe(df: "pd.DataFrame") -> "pd.DataFrame":
    """Make every object column encodable, by casting mixed ones to string.

    `renovation_years` holds "1985; 2018" on one report and 1985 on the next -
    it is free text with no META entry, so coerce_types never normalises it,
    and pandas hands Arrow an object column containing both str and int.
    Arrow refuses ("Expected bytes, got a 'int' object") and the parquet write
    dies. This has now bitten twice: once taking a whole 132-report run's
    outputs with it, and once here, silently.

    Casting only the genuinely mixed columns, not every object column, so a
    clean string column keeps its nulls as NaN rather than the string "nan".
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        kinds = {type(v) for v in out[col] if v is not None and v == v}
        if len(kinds) > 1:
            out[col] = out[col].map(lambda v: v if v is None or v != v else str(v))
    return out


def write_outputs(tables: dict, s3_partitions: bool = True):
    AGG.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(AGG / f"{name}.csv", index=False)
        if name != "manifest":
            target = AGG / f"{name}.parquet"
            try:
                _parquet_safe(df).to_parquet(target, index=False)
            except Exception as e:
                # A stale parquet beside a fresh CSV is worse than no parquet:
                # the two disagree and nothing says so. This is not
                # hypothetical - a failed write here once left a 4-row pilot
                # parquet next to a 131-row CSV. Remove it and say so.
                if target.exists():
                    target.unlink()
                print(f"  (parquet FAILED for {name}, removed stale file so "
                      f"it cannot be mistaken for current: {e})")

    if not s3_partitions or tables["properties"].empty:
        return
    # S3-style layout: partitioning by firm and state makes leave-one-firm-out
    # and leave-one-region-out CV a prefix filter instead of an in-memory one.
    root = AGG / "s3"
    # REBUILD, don't accumulate. Partition paths are derived from the firm and
    # state slugs, so when a firm's normalised name changes - "Terracon
    # Consultants, Inc." folding to "Terracon" - the old partition is not
    # overwritten, it is simply orphaned, and it keeps serving stale rows to
    # anything that globs the tree. Eleven such files survived this corpus's
    # re-extraction and were nearly shipped alongside fresh CSVs.
    if root.exists():
        shutil.rmtree(root)
    props = tables["properties"]
    keys = props[["property_id", "report_firm", "state"]].copy()
    for name in ["properties", "systems", "components",
                 "capex_by_subcategory"]:
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
    ap.add_argument("--revalidate", action="store_true",
                    help="re-run the free validation stack over the JUDGED "
                         "records in data/cache/ and rebuild the tables. No "
                         "API calls. Use after a validator or vocabulary fix: "
                         "unlike verify_offline.py it reads the post-judge "
                         "record, so the judge's corrections survive.")
    ap.add_argument("--prompt-audit", action="store_true",
                    help="report which prompt version produced each cached "
                         "extraction and exit. 'Extracted under one frozen "
                         "prompt' is the dataset's headline claim; this is "
                         "what checks it.")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild tables from cached JSON, no API calls")
    args = ap.parse_args()

    if args.revalidate:
        # Re-validate WITHOUT re-extracting and WITHOUT re-judging.
        #
        # verify_offline.py cannot do this job: it reads data/cache/raw/, which
        # is the extraction BEFORE the judge ran, so every correction the judge
        # made would be silently rolled back. The outer cache holds the record
        # as it stood after judging, which is the one the tables are built
        # from - so a validator fix is applied here, to that record.
        #
        # The judge's own findings are carried forward untouched: `unresolved`
        # is replayed from the cached entry rather than recomputed, because
        # nothing in this pass is entitled to overturn a verdict it did not
        # pay for.
        results, changed = [], 0
        for f in sorted(CACHE.glob("*.json")):
            entry = json.loads(f.read_text())
            record = entry.get("record")
            if not record:
                continue
            before = entry["meta"].get("status")
            repaired = _backfill_join_keys(
                record, entry["meta"].get("property_id"))
            if repaired:
                print(f"  {entry['meta'].get('property_id','?')[:46]:48s} "
                      f"backfilled {repaired} null join key(s)")
            coercions = coerce_types(record)
            det = deterministic_checks(record) + coercions
            recon = (reconciliation_checks(record)
                     + completeness_checks(record))
            nulled = [c for c in coercions if "nulled" in c["detail"]]
            unresolved = entry.get("flags", {}).get("unresolved", [])
            table_issues = (recon + [i for i in det if i["where"] != "property"]
                            + nulled)
            status = ("clean" if not unresolved and not table_issues
                      else "needs_review")
            entry["flags"].update({"deterministic": det, "reconciliation": recon,
                                   "table_issues": table_issues})
            entry["meta"].update({
                "status": status, "det_issues": len(det),
                "recon_issues": len(recon), "revalidated": True,
                "why": "; ".join(sorted({i["kind"] for i in table_issues}))
                       or ("unverified fields" if unresolved else "")})
            f.write_text(json.dumps(entry, indent=2))
            stale = REVIEW / f"{entry['meta']['property_id']}.flags.json"
            if status == "needs_review":
                REVIEW.mkdir(parents=True, exist_ok=True)
                stale.write_text(json.dumps(entry["flags"], indent=2))
            elif stale.exists():
                stale.unlink()
            if before != status:
                changed += 1
                print(f"  {entry['meta']['property_id'][:46]:48s} "
                      f"{before} -> {status}")
            results.append(entry)
        print(f"\nrevalidated {len(results)} report(s); {changed} changed status")

    elif args.prompt_audit:
        import collections
        raw = sorted((CACHE / "raw").glob("*.json"))
        vers = collections.Counter()
        for f in raw:
            try:
                vers[json.loads(f.read_text()).get("_prompt_version")
                     or "unversioned"] += 1
            except Exception:
                vers["unreadable"] += 1
        print(f"current prompt version: {PROMPT_VERSION}\n")
        print(f"{len(raw)} cached extraction(s):")
        for v, n in vers.most_common():
            mark = "  <- current" if v == PROMPT_VERSION else ""
            print(f"  {v:14s} {n:4d}{mark}")
        n_unver = vers.get("unversioned", 0)
        n_cur = vers.get(PROMPT_VERSION, 0)
        if n_unver:
            # "unversioned" is not a version - it is the ABSENCE of one, and
            # it must never be reported as uniformity. The handoff records
            # that these 134 records came from roughly six prompt versions
            # over one session; the stamp did not exist to distinguish them.
            print(f"\nUNKNOWN PROVENANCE: {n_unver} extraction(s) carry no "
                  f"prompt stamp. That is not evidence they share a prompt - "
                  f"it is the absence of evidence either way, and the handoff "
                  f"records these as the product of ~6 prompt versions run in "
                  f"firm-grouped batches. Treat them as a firm-correlated "
                  f"confound on the leave-one-firm-out CV axis until "
                  f"re-extracted.")
        if len(vers) > 1:
            print(f"\nMIXED - {len(vers)} distinct provenances. Re-run to "
                  f"bring every report onto {PROMPT_VERSION}; reports already "
                  f"matching it are reused for free.")
        elif n_cur:
            print(f"\nUNIFORM - all {n_cur} extraction(s) came from the "
                  f"current prompt {PROMPT_VERSION}. The frozen-prompt claim "
                  f"holds for this dataset.")
        return

    elif args.aggregate_only:
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
    print(f"capex table: {len(tables['capex_by_subcategory'])} rows "
          f"(property x 12 subcategories - the modelling grain)")
    tq = tables["properties"]["target_quality"].value_counts().to_dict()
    print(f"target_quality: {tq}   <- filter capex training on this, "
          f"not on extraction_status")
    if not tables["properties"].empty:
        firms = tables["properties"]["report_firm"].value_counts().to_dict()
        print(f"firms:       {firms}")
    print(f"\nwritten to {AGG}")
    # A normal run aggregates only the reports it just processed. Any report
    # that has a validated record but no PDF in the folder - three of this
    # corpus were delivered without their source documents - would otherwise
    # vanish from the training set without a word. Say so; do not silently
    # ship a smaller dataset than exists.
    if not args.aggregate_only:
        processed = {m.get("property_id") for m in
                     (r["meta"] for r in results)}
        absent = sorted({f.stem for f in CACHE.glob("*.json")} - processed)
        if absent:
            print(f"\n{len(absent)} validated report(s) were NOT part of this "
                  f"run because no PDF for them is in the folder:")
            for a in absent:
                print(f"    {a}")
            print("  They are missing from the tables just written. To build "
                  "the full dataset including them:\n"
                  "      python batch.py --aggregate-only --include-flagged\n"
                  "  They carry prompt_version='unversioned' so you can filter "
                  "them out of any analysis that needs one frozen prompt.")

    n_flagged = counts.get("needs_review", 0)
    n_error = counts.get("error", 0)
    if args.include_flagged:
        # They are IN the tables - saying "excluded" here was simply wrong and
        # is the kind of line someone quotes in a methods section.
        if n_flagged:
            print(f"\n{n_flagged} needs_review report(s) ARE INCLUDED in the "
                  f"tables, tagged extraction_status='needs_review'. Filter on "
                  f"that column for any analysis where the financial totals "
                  f"must tie out; their flags are in data/needs_review/.")
        if n_error:
            print(f"{n_error} report(s) errored and are NOT in the tables. "
                  f"Re-run this same command to retry only those.")
    elif n_flagged or n_error:
        print(f"\n{n_flagged + n_error} report(s) excluded from the tables. "
              f"Review their flags in data/needs_review/ before treating the "
              f"aggregate as complete - silently dropping failures biases the "
              f"training set.")


if __name__ == "__main__":
    main()