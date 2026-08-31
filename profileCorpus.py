#!/usr/bin/env python
"""
Survey a folder of PCA PDFs BEFORE committing to a full extraction run.

    uv run python profile_corpus.py ~/path/to/all_pdfs
    uv run python profile_corpus.py ~/path/to/all_pdfs --workers 8

Reads no API. Costs nothing. Takes roughly 10-30 minutes on 135 reports,
because it extracts the text layer of every page to find where the cost
tables actually live - the same scan extract.py does, using the same markers,
so the answer reflects what the pipeline will really send.

Writes corpus_profile.csv and prints a summary answering the questions that
decide whether a full run is worth starting:

  * How many DISTINCT FIRMS are in there? Every firm so far has had its own
    table shape, and each one needed a prompt change. Firms not in the four
    already handled are the real risk, not page counts.
  * Do any reports have NO TEXT LAYER? Those are scans. Page selection falls
    back to the front block, and grounding cannot verify anything, so they
    will behave differently from everything tested so far.
  * Do the cost tables FIT THE PAGE BUDGET? A report whose table pages do not
    fit loses rows silently - reconciliation catches it as an under-sum, but
    only after you have paid for the extraction.
  * Are there DUPLICATES? Same property twice inflates the training set with
    correlated rows and quietly breaks any leave-one-property-out CV.
"""
import argparse
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

# Use the pipeline's own markers and budget so this profile predicts what the
# pipeline will do, rather than approximating it.
from extract import _TABLE_MARKERS, FRONT_PAGES, API_MAX_PAGES, _structural_score
from schema import MAX_PDF_PAGES, FIRM_PATTERNS, CLIENT_NAMES

# Firms and their canonical names now live in schema.py, so the profiler and
# the pipeline cannot drift apart on the question of who wrote a report - a
# firm the profiler calls "Nova" and the extractor calls "Nova Consulting"
# would partition as two firms in the S3 layout and quietly leak across a
# leave-one-firm-out split. KNOWN_FIRMS is the subset whose table shape the
# extractor has actually been written against; everything else is named but
# still flagged as new.
_FIRM_LIST = [(pat, name) for pat, name in FIRM_PATTERNS if name != "OTHER"]

HANDLED_SHAPES = {
    "EBI Consulting", "EMG", "Bureau Veritas", "Partner Engineering",
    "AEI", "Nova", "NV5", "Lender Consulting", "EPIC", "Terracon",
    "Gabion", "Tetra Tech", "LandScience", "CBC", "Atwell",
    "Metropolitan Solutions",
}

# Clients, owners and lenders. These appear prominently on cover pages
# ("Prepared for: Lloyd Jones LLC") and will otherwise be mistaken for the
# assessing firm - which is what happened on the first pass.
CLIENT_NOISE = re.compile(CLIENT_NAMES, re.I)

# A real cost-table header. Marker density alone picks prose pages that happen
# to mention several keywords; this is the signature of an actual table.
def _is_header_page(low: str) -> bool:
    """Does this page carry an actual cost-table header?

    The original test demanded EUL and RUL, which is a shape-A assumption.
    Gabion's table has neither ("Present Worth / Start Year of Occurrence"),
    Terracon's has EUL but no RUL, and LandScience and CBC cost their work in
    a section table with no life columns at all - so on those firms the
    profiler ranked a narrative page as the best "table" and the sample it
    printed was prose. Any one of these header signatures is enough.
    """
    life = ("eul" in low or "expected life" in low
            or "expected useful life" in low)
    money_col = ("unit cost" in low or "total cost" in low
                 or "present worth" in low or "r-total" in low
                 or "immed. cost" in low or "reserve cost" in low)
    if life and money_col:
        return True
    if "capital considerations" in low and "present worth" in low:
        return True
    if "reflective age" in low or "overall expected life" in low:
        return True
    return False


# Column names, used to score how table-like a page is. A narrative section
# discussing remaining useful life mentions two or three of these; an actual
# cost table header carries most of them on one line.
_COLUMN_TOKENS = (
    "eff age", "eul", "rul", "quantity", "unit cost", "cycle replace",
    "replace percent", "on site qty", "qty in eval", "base cost",
    "total cost", "year 1", "yr 1", "description", "item", "sect",
    # columns the firms found in the wider corpus use instead
    "present worth", "work span", "of occurr", "start year", "class",
    "expected life", "remaining life", "reflective age", "r-total",
    "immed. cost", "reserve cost", "unit price", "cumulative",
)


def _table_score(low: str) -> tuple:
    """(distinct column names, digit density). Ranks a real table above prose.

    Ranking header pages by length instead picks the longest narrative page
    that happens to say EUL and total cost, which is exactly what went wrong
    on the first pass and left most firms' samples unreadable.
    """
    cols = sum(1 for t in _COLUMN_TOKENS if t in low)
    digits = sum(c.isdigit() for c in low) / max(1, len(low))
    return (cols, digits)


# Column signatures that decide what code is needed. Firms matter less than
# shapes: several "new" firms turn out to emit a table already handled.
# Ordered most specific first: several firms carry more than one of these
# tokens, and the first match wins.
# Ordered by how DISTINCTIVE the signature is, not by shape letter. The first
# match wins, so a phrase that only one layout uses has to be tested before a
# phrase several use in passing: "reserve cost" appears in the prose of most
# reports, while "qty in eval" appears in exactly one firm's table header.
#
# LITERAL PHRASES, not regexes. PDF text layers drop spaces unpredictably -
# "Cycle Replace" in one file is "CycleReplace" in the next, and NV5 emits
# "UnitCost" - so each phrase is also matched with its punctuation and spacing
# removed. That only works on literals: squashing a regex silently rewrites
# it. "r-total|cumulative r ?-" squashed down to "rtotal|cumulativer", and
# "rtotal" is a substring of "yeartotal", which put 23 reports into Terracon's
# two-report shape.
TABLE_SHAPES = [
    (("qty in eval", "on site qty"),         "B: Partner (qty-in-eval)"),
    (("capital considerations", "present worth", "work span"),
                                             "E: Gabion (capital considerations)"),
    (("r-total$", "r - total", "cumulative r -"),
                                             "F: Terracon (R-numbered)"),
    # AEI's Homewood Suites set wraps its header across two lines, so the
    # column names never appear contiguously: the page reads "Overall Expected
    # Remaining ... Life (a) Life (b)". Match the fragments that survive that.
    (("reflective age", "overall expected life", "overall expected remaining",
      "life (a) life (b)"),                  "I: spelled-out life columns"),
    (("replace percent", "cycle replace"),   "A: EBI/BV (replace percent)"),
    (("base cost",),                         "C: Nova (base cost)"),
    (("rating 1-5", "rul:eul"),              "D: EMG (numeric 1-5 rating)"),
    (("cost summary recommendation",),       "G: inline per-section summaries"),
    (("immed. cost", "immediate need repair", "probable immediate repairs"),
                                             "H: narrative-costed"),
]


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text)


# Squashing removes the word boundaries along with the spaces, so a short
# squashed phrase turns into a substring of ordinary words: "r-total$" squashes
# to "rtotal", which lives inside "yeartotal" and put 21 reports that belong to
# other shapes into Terracon's. Below this length a phrase is matched only in
# its spaced form, where the boundaries still exist.
_MIN_SQUASH_LEN = 8


def _classify_shape(doc: str, squashed_doc: str) -> str:
    for phrases, label in TABLE_SHAPES:
        for phrase in phrases:
            if phrase in doc:
                return label
            sq = _squash(phrase)
            if len(sq) >= _MIN_SQUASH_LEN and sq in squashed_doc:
                return label
    return "UNCLASSIFIED"


def _text(page) -> str:
    try:
        return (page.extract_text() or "")
    except Exception:
        return ""


def _flat(s: str, limit: int) -> str:
    """One-line, length-capped excerpt that survives a CSV round trip."""
    return re.sub(r"\s+", " ", (s or "")).strip()[:limit]


def real_pdfs(folder: Path) -> list:
    """Every *.pdf under folder, minus macOS archive litter.

    Unzipping a Mac-made archive leaves __MACOSX/._Report.pdf AppleDouble
    stubs beside the real files. They match *.pdf and they are not PDFs -
    they are a few KB of resource-fork metadata. Left in, they double the
    apparent report count here, and in batch.py they would each be handed to
    the extractor at ~$1-2 per attempt before failing.
    """
    return sorted(p for p in folder.rglob("*.pdf")
                  if not p.name.startswith("._")
                  and "__MACOSX" not in p.parts)


def profile_one(path_str: str) -> dict:
    path = Path(path_str)
    out = {"file": path.name, "pages": None, "mb": round(path.stat().st_size / 1e6, 1),
           "firm": "UNKNOWN", "firm_known": False, "text_pages_pct": None,
           "marker_pages": 0, "first_marker": None, "last_marker": None,
           "front_covers_tables": None, "fits_budget": None, "pages_to_send": None,
           "sha1_first_page": None, "cover_sample": "", "table_page": None,
           "has_header": False, "table_shape": "UNCLASSIFIED",
           "eval_period": None, "firm_pages": 0,
           "table_sample": "", "risk_flags": ""}
    flags = []
    try:
        reader = PdfReader(path_str)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                out["risk_flags"] = "ENCRYPTED"
                return out
        pages = reader.pages
        out["pages"] = len(pages)

        texts = [_text(p) for p in pages]
        lowered = [t.lower() for t in texts]

        with_text = sum(1 for t in texts if len(t.strip()) > 50)
        out["text_pages_pct"] = round(100 * with_text / max(1, len(pages)))
        if out["text_pages_pct"] < 20:
            flags.append("NO_TEXT_LAYER(scanned?)")

        # Firm detection across the WHOLE document, not the cover.
        # Assessment firms stamp their name in the page footer and in project
        # numbers on every page; the client's name appears once, on the cover.
        # Scanning only the front therefore finds the client - which is how 58
        # reports came back UNKNOWN while naming Lloyd Jones, their owner.
        scored = {}
        for low in lowered:
            clean = CLIENT_NOISE.sub(" ", low)
            for pat, name in _FIRM_LIST:
                if re.search(pat, clean):
                    scored[name] = scored.get(name, 0) + 1
        if scored:
            best = max(scored.items(), key=lambda kv: kv[1])
            out["firm"] = best[0]
            out["firm_pages"] = best[1]
            out["firm_known"] = best[0] in HANDLED_SHAPES
        if out["firm"] == "UNKNOWN":
            flags.append("FIRM_UNRECOGNISED")
        elif not out["firm_known"]:
            flags.append("NEW_FIRM")

        # Same scoring the extractor uses, markers plus structure, so the
        # profile predicts the real slice rather than approximating it.
        marker_pages = [i for i, t in enumerate(lowered)
                        if sum(1 for m in _TABLE_MARKERS if m in t)
                        + _structural_score(t) >= 2]
        out["marker_pages"] = len(marker_pages)
        if marker_pages:
            out["first_marker"] = marker_pages[0] + 1
            out["last_marker"] = marker_pages[-1] + 1
            out["front_covers_tables"] = marker_pages[-1] < FRONT_PAGES
        else:
            flags.append("NO_TABLE_MARKERS")

        budget = min(MAX_PDF_PAGES, API_MAX_PAGES)
        if len(pages) <= budget:
            out["pages_to_send"], out["fits_budget"] = len(pages), True
        else:
            front = set(range(min(FRONT_PAGES, budget)))
            tail = set()
            for i in marker_pages:
                tail.update(j for j in (i - 1, i, i + 1)
                            if 0 <= j < len(pages) and j not in front)
            room = budget - len(front)
            out["pages_to_send"] = len(front) + min(len(tail), room)
            out["fits_budget"] = len(tail) <= room
            if not out["fits_budget"]:
                flags.append(f"TABLES_EXCEED_BUDGET(need {len(front)+len(tail)})")

        if texts:
            out["sha1_first_page"] = hashlib.sha1(
                texts[0].strip().encode("utf-8", "ignore")).hexdigest()[:12]
            out["cover_sample"] = _flat(texts[0], 400)

        # WHERE the tables are tells you whether the run will work; WHAT SHAPE
        # they are tells you what code to write. Prefer a page carrying a real
        # header signature over the densest marker page - ranking by marker
        # count alone lands on cover letters and narrative sections that happen
        # to say "EUL" and "total cost", which is why most firms' samples came
        # back as prose on the first pass.
        header_pages = [i for i in marker_pages if _is_header_page(lowered[i])]
        pick = None
        if header_pages:
            pick = max(header_pages, key=lambda i: _table_score(lowered[i]))
        elif marker_pages:
            pick = max(marker_pages, key=lambda i: _table_score(lowered[i]))
        if pick is not None:
            out["table_page"] = pick + 1
            out["has_header"] = bool(header_pages)
            # Start the excerpt at the header itself, not the page top, so the
            # column names survive the length cap.
            low = lowered[pick]
            j = low.find("eul")
            start = max(0, j - 120) if j != -1 else 0
            out["table_sample"] = _flat(texts[pick][start:start + 1600], 1500)

        doc = " ".join(lowered)
        # PDF text layers drop spaces unpredictably: the same column header
        # renders as "Cycle Replace" in one firm's file and "CycleReplace" in
        # another's, and NV5 emits "UnitCost" and "TotalCost". Matching only
        # the spaced form left AEI's twelve reports and EBI's eight sitting in
        # UNCLASSIFIED while carrying a textbook shape-A table. Test both
        # forms, the way extract.py already does for its page markers.
        out["table_shape"] = _classify_shape(doc, _squash(doc))
        if re.search(r"year 12|yr 12", doc):
            out["eval_period"] = 12
        elif re.search(r"year 10|yr 10", doc):
            out["eval_period"] = 10
    except Exception as e:
        flags.append(f"UNREADABLE:{type(e).__name__}")

    out["risk_flags"] = "; ".join(flags)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="corpus_profile.csv")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    pdfs = real_pdfs(folder)
    if not pdfs:
        zips = list(folder.rglob("*.zip"))
        print(f"No PDFs under {folder}")
        if zips:
            print(f"\nFound {len(zips)} zip file(s) there. Unpack first:\n")
            print(f'  cd "{folder}"')
            print('  for z in *.zip; do unzip -q -o "$z" -d "${z%.zip}"; done')
            print('  find . -name "__MACOSX" -type d -exec rm -rf {} +')
            print('  find . -name "._*" -delete')
        return
    print(f"Profiling {len(pdfs)} PDFs with {args.workers} workers "
          f"(no API calls)...\n")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(profile_one, str(p)): p for p in pdfs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(pdfs)}] {str(r['pages']):>4}p  {r['mb']:>6}MB  "
                  f"{r['firm'][:22]:22} {r['risk_flags'][:60]}")

    df = pd.DataFrame(rows).sort_values("file")
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 70)
    print(f"{len(df)} reports profiled -> {args.out}\n")

    print("FIRMS")
    for firm, cnt in df["firm"].value_counts().items():
        known = df[df["firm"] == firm]["firm_known"].iloc[0]
        print(f"  {firm[:30]:32} {cnt:4}  {'handled' if known else 'NEEDS WORK'}")

    unhandled = df[~df["firm_known"]]
    print(f"\n  reports from firms the extractor has NOT been built for: "
          f"{len(unhandled)} of {len(df)}")

    print("\nPAGES")
    p = df["pages"].dropna()
    if len(p):
        print(f"  min {int(p.min())}  median {int(p.median())}  "
              f"max {int(p.max())}  total {int(p.sum()):,}")
        sent = df["pages_to_send"].dropna()
        print(f"  pages actually sent per report: "
              f"median {int(sent.median())}, total {int(sent.sum()):,}")

    print("\nRISKS")
    for label, mask in [
        ("no text layer (scanned)",      df["risk_flags"].str.contains("NO_TEXT", na=False)),
        ("no cost-table markers found",  df["risk_flags"].str.contains("NO_TABLE_MARKERS", na=False)),
        ("tables exceed page budget",    df["fits_budget"] == False),
        ("unreadable / encrypted",       df["risk_flags"].str.contains("UNREADABLE|ENCRYPTED", na=False)),
        ("firm unrecognised",            df["firm"] == "UNKNOWN"),
    ]:
        hit = df[mask]
        print(f"  {label:32} {len(hit):4}")
        for f in hit["file"].head(5):
            print(f"      {f[:66]}")

    dupes = df[df["sha1_first_page"].notna() & df["sha1_first_page"].duplicated(keep=False)]
    print(f"  {'duplicate first pages':32} {len(dupes):4}")
    for f in dupes["file"].head(6):
        print(f"      {f[:66]}")

    n = len(df)
    print("\nFULL-RUN ESTIMATE (from your four measured reports: 153s, 188s, "
          "1732s, 1999s)")
    print(f"  cost   ~${n * 1.0:,.0f} - ${n * 2.0:,.0f}")
    for w in (3, 6):
        print(f"  time   ~{n * 1018 / w / 3600:.1f} h at {w} workers "
              f"(median-case; senior-housing reports ran 10x the others)")
    print("\nTABLE SHAPE (what the extractor actually keys on - firms matter "
          "less than shapes)")
    for shape, cnt in df["table_shape"].value_counts().items():
        print(f"  {shape[:44]:46} {cnt:4}")
    known_shape = df["table_shape"].astype(str).str.match(r"[A-D]:").sum()
    print(f"\n  reports whose table shape is already handled: "
          f"{known_shape} of {len(df)}")
    print(f"  reports with a real header captured:          "
          f"{int(df['has_header'].sum())} of {len(df)}")
    ep = df["eval_period"].value_counts().to_dict()
    print(f"  evaluation period (years): {ep}")

    print("\nHEADER SAMPLES (one per firm - this is what drives code changes)")
    for firm, grp in df[df["table_sample"].astype(str).str.len() > 0].groupby("firm"):
        row = grp.iloc[0]
        tp = row["table_page"]
        fname = str(row["file"])[:44]
        tpage = int(tp) if pd.notna(tp) else "?"
        sample = str(row["table_sample"])[:340]
        print(f"\n  --- {firm}  (n={len(grp)}, e.g. {fname} p{tpage})")
        print(f"      {sample}")

    print("\n  Send corpus_profile.csv. The table_sample column carries each "
          "firm's actual\n  column headers, which is what the extractor's "
          "field mapping is written from.\n  Only send a PDF if a firm's "
          "sample comes back empty or unreadable.")


if __name__ == "__main__":
    main()