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

v3 changes, all aimed at the same failure: reports whose money reconciles
exactly being excluded from the training set over descriptive fields.

  _num          parsed by concatenating every digit in the string, so
                "44,495 (per the rent roll dated August 02, 2023)" became
                44495022023. Now takes the first well-formed number.
  coerce_types  reported SUCCESSFUL coercions ("444 units" -> 444) as issues,
                so a correct recovery flagged the report. Now only genuine
                losses are returned; recoveries go to stderr.
                Also parses ranges ("1-4", "One and two floors") into
                min/max fields instead of nulling the field, and rewrites
                firm-specific condition spellings onto the canonical scale.
  grounding     compared snippets to pdfplumber's text layer with only
                whitespace folding. The model reads the PDF through a
                different extractor, so spacing, hyphens, accents and dropped
                spaces ("wasobserved") broke exact matching on text that was
                plainly present. Now: unicode folding, ellipsis-aware
                fragment matching, and a punctuation-stripped fallback.
                Also stopped re-joining and re-normalising the whole document
                once per property field - that was ~50 passes over a 177-page
                report per validation.

Blocking policy (which flags route a report to needs_review) lives in the
pipeline, not here. This module only decides what is true.
"""
import re
from pathlib import Path
import sys
import unicodedata

import pdfplumber

from schema import (PROPERTY_FIELDS, PROPERTY_META, SYSTEM_META, CONDITIONS,
                    COMPONENT_META, RECONCILE_REL_TOL, CONFIDENCE_FLOOR,
                    RANGE_NUMBER_FIELDS, CATEGORY_SYNONYM_FIELDS,
                    RANGE_COMPONENT_FIELDS, RESERVE_YEARS, YEAR_FIELDS,
                    FIRM_PATTERNS, CLIENT_NAMES, NEAR_TERM_TABLES,
                    US_STATES)


# ── number parsing ─────────────────────────────────────────────────────────

# A number with optional thousands separators and decimals. Leading "-" only
# when it is not acting as a range dash ("1-4" must read as 1, not -4).
_NUM_TOKEN = re.compile(r"(?<![\d-])-?\d[\d,]*(?:\.\d+)?")

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    # Counts written as adjectives: "Single story", "Two buildings".
    "single": 1, "double": 2, "triple": 3,
}

# "1-4", "1 to 4", "2 - 3 stories"
_DIGIT_RANGE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:-|–|—|to|through|thru|and)\s*(\d[\d,]*(?:\.\d+)?)",
    re.I)


def _all_numbers(x: str) -> list:
    out = []
    for tok in _NUM_TOKEN.findall(str(x)):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _num(x):
    """First well-formed number in x, or None.

    Deliberately NOT "every digit in the string glued together": that turned
    parenthetical dates and rent-roll citations into eleven-digit garbage that
    only the range checks caught, and only sometimes.
    """
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return None
    nums = _all_numbers(x)
    if nums:
        return nums[0]
    # No digits: fall back to a spelled-out number. Reports write "One" into
    # num_buildings and "Single story" into num_stories, and nulling those
    # discarded a real value AND flagged the report for stating it. Only when
    # the string carries EXACTLY one number word, so "One, and attached
    # parking garage" resolves to 1 while a sentence containing several counts
    # stays null rather than having one picked at random.
    words = _word_numbers(x)
    return float(words[0]) if len(words) == 1 else None


def _word_numbers(x: str) -> list:
    toks = re.findall(r"[a-z]+", str(x).lower())
    return [_WORD_NUMBERS[t] for t in toks if t in _WORD_NUMBERS]


def _number_range(x):
    """(lo, hi) when x states a range of values, else None.

    Conservative on purpose: only an explicit digit range ("1-4", "1 to 4") or
    a string whose numbers are spelled as words ("One and two floors"). A
    string that merely happens to contain several numbers - a measurement with
    a date cited after it - is NOT a range and must not be read as one.
    """
    if not isinstance(x, str):
        return None

    m = _DIGIT_RANGE.search(x)
    if m:
        try:
            a = float(m.group(1).replace(",", ""))
            b = float(m.group(2).replace(",", ""))
        except ValueError:
            return None
        return (min(a, b), max(a, b))

    if not _all_numbers(x):
        words = _word_numbers(x)
        if len(words) >= 2:
            return (float(min(words)), float(max(words)))
        if len(words) == 1:
            return (float(words[0]), float(words[0]))
    return None


# ── text normalisation for grounding ───────────────────────────────────────

_PUNCT_FOLD = {
    **{ord(c): "-" for c in "‐‑‒–—―"},
    **{ord(c): "'" for c in "’‘‚‛"},
    **{ord(c): '"' for c in "“”„‟"},
    ord(" "): " ",
}


def _norm(s: str) -> str:
    """Whitespace-folded, unicode-folded, lowercase."""
    s = unicodedata.normalize("NFKC", str(s)).translate(_PUNCT_FOLD)
    return re.sub(r"\s+", " ", s).strip().lower()


def _squash(s: str) -> str:
    """Letters and digits only, accents stripped.

    The model and pdfplumber parse the PDF with different extractors. They
    disagree about spaces inside table cells, hyphenation at line breaks,
    accented characters, and occasionally about whether a space exists at all
    ("wasobserved"). Squashing removes every axis they can disagree on while
    keeping a 12-word phrase specific enough that a false match is implausible.
    """
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)


def _fragments(snippet: str) -> list:
    """Split a snippet on ellipses.

    The extractor sometimes abbreviates a long quote: "UNINFLATED TOTALS: ...
    $196,940". That can never match verbatim, so each side is matched
    separately and in order.
    """
    parts = re.split(r"\.{2,}|…", str(snippet))
    return [p for p in (p.strip() for p in parts) if p]


def _contains_ordered(haystack: str, needles: list) -> bool:
    pos = 0
    for n in needles:
        if not n:
            continue
        i = haystack.find(n, pos)
        if i == -1:
            return False
        pos = i + len(n)
    return True


def _snippet_present(snippet: str, norm_hay: str, squash_hay: str) -> bool:
    frags = _fragments(snippet)
    if not frags:
        return False
    if _contains_ordered(norm_hay, [_norm(f) for f in frags]):
        return True
    return _contains_ordered(squash_hay, [_squash(f) for f in frags])


# ── layer 1: deterministic ─────────────────────────────────────────────────

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
        # some firms tick multiple condition boxes -> "fair; poor"
        parts = [p.strip().lower() for p in str(value).split(";") if p.strip()]
        bad = [p for p in parts if p not in meta["allowed"]]
        if bad:
            issues.append({"where": where, "field": field, "kind": "category",
                           "detail": f"{bad} not in {meta['allowed']}"})
    elif t == "date":
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(value)):
            issues.append({"where": where, "field": field, "kind": "date",
                           "detail": f"bad date: {value!r}"})
    return issues


_CLIENT_RE = re.compile(CLIENT_NAMES, re.I)


def _client_named_as_firm(prop: dict) -> list[dict]:
    """report_firm holding a client's name is a silent, expensive error.

    The assessing firm is the S3 partition key and the unit of the
    leave-one-firm-out split that the whole generalisation argument rests on.
    Nothing else in the stack can tell "AEI" from "Bridge House Advisors
    Corp." - both are companies, both are strings, and the value passes every
    other check. Raising it as a PROPERTY issue puts report_firm in front of
    the judge, which re-reads the cover; if the judge cannot settle it the
    field stays unresolved and the report goes to review rather than into the
    tables under the wrong firm.
    """
    cell = prop.get("report_firm") or {}
    val = cell.get("value")
    if not isinstance(val, str) or not val.strip():
        return []
    if not _CLIENT_RE.search(val):
        return []
    client = (prop.get("client_name") or {}).get("value")
    return [{"where": "property", "field": "report_firm", "kind": "firm",
             "detail": f"report_firm {val!r} matches a known client/lender, "
                       f"not an assessing firm (client_name={client!r}). "
                       f"Likely read 'Prepared for' instead of 'Prepared by'."}]


def deterministic_checks(record: dict) -> list[dict]:
    issues = []
    prop = record["property"]
    issues += _client_named_as_firm(prop)

    for field in PROPERTY_FIELDS:
        meta = PROPERTY_META.get(field, {})
        cell = prop.get(field) or {}
        issues += _meta_issues(cell.get("value"), meta, field, "property")

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


# ── layer 2: coercion and normalisation ────────────────────────────────────

_SPAN = re.compile(r"\s+(?:to|-|\u2013|/)\s+|\s*/\s*", re.I)
_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")


def _strip_qualifier(part: str, synonyms: dict) -> str:
    """"fair (PVC roof)" -> "fair".

    A rating with a parenthetical naming WHICH component it applies to is a
    correct reading of the report - Tetra Tech rates a roof "fair (PVC)" and
    "poor (EPDM)" where a building has both. The qualifier is real
    information, but it is not part of the RATING, and leaving it attached
    failed the category check and took the report to review over a value the
    model got right.

    Only strips when what remains is a rating we recognise, so "poor (see
    section 4)" loses its parenthetical while a value that is nothing but a
    parenthetical is left alone to fail loudly.
    """
    stripped = _QUALIFIER.sub("", part).strip()
    if not stripped or stripped == part:
        return part
    for piece in _SPAN.split(stripped):
        piece = piece.strip().lower()
        if piece and synonyms.get(piece, piece) not in CONDITIONS:
            return part
    return stripped


def _split_spanning(part: str, synonyms: dict) -> list:
    """"Good to Fair" -> ["Good", "Fair"].

    Firms without tick-box tables write a spanning rating in prose: "good to
    fair", "fair/poor". The schema already has a place for that - condition
    plus condition_secondary - but a single string never reached it, so the
    whole value failed the category check and took the report with it.

    Split only when BOTH halves are recognisable ratings. "roof to be
    replaced" must stay one string, and a rating we do not know should fail
    the category check loudly rather than be quietly torn in half.
    """
    part = _strip_qualifier(part, synonyms)
    halves = [h.strip() for h in _SPAN.split(part) if h and h.strip()]
    if len(halves) != 2:
        return [part]
    canon = [synonyms.get(h.lower(), h.lower()) for h in halves]
    if all(c in CONDITIONS for c in canon):
        return halves
    return [part]


_BARE_RATING = re.compile(r"^\s*([1-5])(?:\s*(?:/|of|out of)\s*5)?\s*$")


def _rehome_numeric_conditions(record: dict) -> None:
    """Move a bare 1-5 condition into condition_rating_numeric.

    Firms that rate numerically (EMG) put a digit where a word goes. The digit
    is real data, but it is not a member of CONDITIONS, so it failed the
    category check and took the whole report to review. The extractor is now
    told to file it correctly, but that only helps NEW extractions - cached
    ones already hold the digit, and re-extracting a report to move one value
    between columns is not a good trade. Handled here so both recover.

    The number is NOT translated into a word. See the note on
    condition_rating_numeric in schema.py: scale direction differs by firm and
    these reports print no legend.
    """
    for row in record.get("systems") or []:
        for field in ("condition", "condition_secondary"):
            val = row.get(field)
            if not isinstance(val, str):
                continue
            m = _BARE_RATING.match(val)
            if not m:
                continue
            row[field] = None
            if field == "condition" and row.get("condition_rating_numeric") is None:
                row["condition_rating_numeric"] = float(m.group(1))
                print(f"[coerce] systems.condition {val!r} -> "
                      f"condition_rating_numeric={m.group(1)}", file=sys.stderr)


def _normalise_categories(record: dict) -> None:
    """Rewrite firm spellings onto the canonical scale, in place.

    Partner Engineering writes "not applicable" where EBI ticks the NA box.
    Same rating, different words. Only exact synonyms are rewritten; a term
    that means something genuinely different (Partner's "functional") is left
    alone and carried in CONDITIONS as its own value.
    """
    for layer, fields in CATEGORY_SYNONYM_FIELDS.items():
        if not fields:
            continue
        if layer == "property":
            rows = [record["property"]]
        else:
            rows = record.get(layer) or []
        for row in rows:
            for field, synonyms in fields.items():
                cell = row.get(field)
                is_cell = isinstance(cell, dict)
                val = cell.get("value") if is_cell else cell
                if not isinstance(val, str):
                    continue
                parts = [p.strip() for p in val.split(";") if p.strip()]
                parts = [q for p in parts for q in _split_spanning(p, synonyms)]
                new = [synonyms.get(p.lower(), p.lower()) for p in parts]
                if not new:
                    continue
                joined = "; ".join(new)
                if joined != val:
                    if is_cell:
                        cell["value"] = joined
                    else:
                        row[field] = joined


# Values a report writes into a numeric cell to say "there is no number
# here". These are STATEMENTS, not failures: "Varies" on a RUL column is the
# firm telling you the component's remaining life differs across the property,
# which the schema already carries as rul_varies. Tetra Tech writes it on a
# quarter of its rows. Treating them as coercion losses would flag the report
# and drop it out of the aggregate for saying something true.
_NON_VALUES = re.compile(
    r"^\s*(?:var|varies|various|n/?a|n\.a\.|na|none|nil|tbd|n/?r|"
    r"not applicable|not assessed|-+|\u2014+)\s*$", re.I)

_LIFE_FIELDS = ("eul_years", "rul_years", "effective_age_years")


def _coerce_row_field(row, field, meta, where, ranges, losses):
    """Coerce one numeric cell on a systems/components row, in place."""
    val = row.get(field)
    if val is None or isinstance(val, (int, float)) or isinstance(val, bool):
        return
    if not isinstance(val, str):
        return

    if _NON_VALUES.match(val):
        row[field] = None
        if field in _LIFE_FIELDS and row.get("rul_varies") is not True:
            # Only "varies" says the life differs; "n/a" says there is no
            # life to state. Do not conflate them.
            if re.match(r"^\s*var", val, re.I):
                row["rul_varies"] = True
        return

    rng = _number_range(val) if field in ranges else None
    if rng is not None:
        lo, hi = rng
        lo_field, hi_field = ranges[field]
        row[lo_field], row[hi_field] = lo, hi
        row[field] = hi
        print(f"[coerce] {where}.{field}: {val!r} -> range {lo:g}-{hi:g} "
              f"({field}={hi:g})", file=sys.stderr)
        return

    n = _num(val)
    row[field] = n
    if n is None:
        losses.append({"where": where, "field": field, "kind": "type_coercion",
                       "detail": f"non-numeric value nulled: {val!r}"})
    else:
        print(f"[coerce] {where}.{field}: {val!r} -> {n:g}", file=sys.stderr)


def _coerce_rows(record: dict) -> list[dict]:
    """Same treatment for the systems and components layers.

    This used to run on the property layer only, which left the two layers
    that actually carry the training data uncoerced: a numeric column arriving
    as a string was flagged by the deterministic type check and then written
    to the CSV as a string anyway. Two things follow from that. Parquet gets
    an object column where a float belongs, and - the expensive one - the
    report is sent to needs_review over a value that was perfectly readable.

    The corpus makes this routine rather than exceptional. Terracon quotes EUL
    as "5-7" and "20-25"; AEI quotes expected life as "10-15" and remaining
    life as "1-2"; several firms write "$1,250.00" into a cost cell and "20
    yrs" into a life cell. Ranges split into min/max the same way num_stories
    does, with the base field taking the max, so a range costs the report
    nothing and the uncertainty stays on the record.
    """
    losses = []
    for layer, metas, ranges in (("systems", SYSTEM_META, {}),
                                 ("components", COMPONENT_META,
                                  RANGE_COMPONENT_FIELDS)):
        for i, row in enumerate(record.get(layer) or []):
            for field, meta in metas.items():
                if meta.get("type") != "number":
                    continue
                _coerce_row_field(row, field, meta, f"{layer}[{i}]",
                                  ranges, losses)
            # Backfill the base field when the model filled min/max directly.
            for field, (lo_f, hi_f) in ranges.items():
                if row.get(field) is None and row.get(hi_f) is not None:
                    row[field] = row[hi_f]
    return losses


_FIRM_RES = [(re.compile(pat, re.I), name) for pat, name in FIRM_PATTERNS]


def _normalise_firm(record: dict) -> None:
    """Fold the extracted report_firm onto a canonical firm name.

    Whatever the model reads off a cover page is whatever that cover page
    says: "EBI Consulting", "EBI", "EBI Consulting, Inc.", "Partner
    Engineering and Science, Inc.", "Partner ESI". Left as-is these are
    distinct strings, and the S3 layout partitions on firm=, so one firm
    becomes four partitions and leave-one-firm-out CV silently leaks - the
    held-out firm is still in the training fold under a different spelling.
    That is the project's headline generalisation claim, so it has to be one
    string per firm.

    Only rewrites when a registered pattern matches. An unrecognised firm
    keeps its extracted name rather than being flattened to UNKNOWN: naming
    it is what makes it countable when it turns up again.
    """
    cell = record["property"].get("report_firm")
    if not isinstance(cell, dict):
        return
    val = cell.get("value")
    if not isinstance(val, str) or not val.strip():
        return
    for rx, name in _FIRM_RES:
        if name != "OTHER" and rx.search(val):
            if val != name:
                print(f"[coerce] property.report_firm: {val!r} -> {name!r}",
                      file=sys.stderr)
                cell["value"] = name
            return


def resolve_firm_from_pdf(record: dict, pdf_path: str) -> bool:
    """Re-derive report_firm from the document when the model read the client.

    The assessing firm stamps its name in the running header or footer of
    nearly every page, along with its project number; the client appears once,
    on the cover, after "Prepared for". So counting name hits ACROSS PAGES
    recovers the assessor even when the cover misleads - the same technique
    profileCorpus.py uses, and it gets all six Homewood Suites reports right
    (AEI) where the extractor returned the lender, "Bridge House Advisors
    Corp.", on every one.

    Runs only when the current value looks like a client, so a correctly
    extracted firm is never second-guessed. Client names are stripped from the
    page text before matching, or the same cover line would vote for them.
    Returns True if it changed anything.
    """
    cell = record["property"].get("report_firm")
    if not isinstance(cell, dict):
        return False
    current = cell.get("value")
    is_client = isinstance(current, str) and bool(_CLIENT_RE.search(current))
    is_blank = not isinstance(current, str) or not current.strip()
    if not (is_client or is_blank):
        return False

    try:
        pages = _page_texts(pdf_path)
    except Exception:
        return False

    scores = {}
    for text in pages:
        low = _CLIENT_RE.sub(" ", text.lower())
        for rx, name in _FIRM_RES:
            if name != "OTHER" and rx.search(low):
                scores[name] = scores.get(name, 0) + 1
    if not scores:
        return False
    best, hits = max(scores.items(), key=lambda kv: kv[1])
    # How much evidence is enough depends on what we are replacing. When the
    # field currently holds a KNOWN CLIENT we already know it is wrong, so any
    # firm named in the document beats a lender's name - AEI is printed on
    # just one page of its Homewood Suites reports, and requiring three left
    # six reports partitioned under "Bridge House Advisors Corp.". When the
    # field is merely blank we have no such certainty, so hold out for the
    # repeated header/footer pattern.
    if best == current or hits < (1 if is_client else 3):
        return False
    print(f"[firm] {Path(pdf_path).name}: report_firm {current!r} -> {best!r} "
          f"(named on {hits} pages)", file=sys.stderr)
    cell["value"] = best
    cell["snippet"] = None
    cell["confidence"] = 0.9
    return True


def _normalise_state(record: dict) -> None:
    """Fold `state` onto its two-letter code, in place. See US_STATES."""
    cell = record["property"].get("state")
    if not isinstance(cell, dict):
        return
    val = cell.get("value")
    if not isinstance(val, str) or not val.strip():
        return
    key = val.strip().lower().rstrip(".")
    code = US_STATES.get(key)
    if code is None and len(key) == 2 and key.upper() in set(US_STATES.values()):
        code = key.upper()
    if code and code != val:
        print(f"[coerce] property.state: {val!r} -> {code!r}", file=sys.stderr)
        cell["value"] = code


def _dedupe_denominators(record: dict) -> None:
    """One count written into two denominator fields is one count, not two.

    Senior-housing reports state a single figure ("96 units") and the
    extractor sometimes files it as num_units AND num_rooms - eight reports in
    this corpus, every one of them identical values on an assisted-living or
    healthcare property. Left alone it double-counts the denominator and makes
    per-unit cost metrics ambiguous.

    Only fires when the values are EQUAL. Genuinely mixed-use properties do
    exist and must survive: King Edward is a 250-unit apartment building AND a
    186-room hotel, and Brentwood at La Porte reports 124 units / 108 rooms.
    Those are two real facts, not a duplication, so they are left alone.
    """
    prop = record["property"]
    basis = (prop.get("unit_basis") or {}).get("value")
    units = _num((prop.get("num_units") or {}).get("value"))
    rooms = _num((prop.get("num_rooms") or {}).get("value"))
    if units is None or rooms is None or units != rooms:
        return
    drop = "num_rooms" if basis != "rooms" else "num_units"
    prop[drop]["value"] = None
    print(f"[coerce] property.{drop}: {units:g} duplicated the other "
          f"denominator (unit_basis={basis!r}) - nulled", file=sys.stderr)


def coerce_types(record: dict) -> list[dict]:
    """Force declared-numeric fields to actually be numbers, and normalise
    firm-specific category spellings.

    Reports say things like num_stories = "One and two floors". That is a true
    statement and the judge will approve it, but it is not a number: it breaks
    Parquet, and silently poisons any numeric feature downstream.

    Three outcomes, and only the third is a problem:
      recovered  "15 stories" -> 15. Correct. Logged to stderr, NOT returned
                 as an issue - a successful recovery used to flag the report,
                 which is the pipeline excluding a report for working.
      range      "1-4" / "One and two floors" -> min/max fields, base field
                 gets the max. A property with two buildings of different
                 heights is a fact about the property, not an extraction
                 failure, and should not cost you the whole report.
      lost       nothing numeric in there at all -> field nulled and the
                 prose returned in the flags, never discarded silently.
    """
    _rehome_numeric_conditions(record)
    _normalise_state(record)
    _dedupe_denominators(record)
    _normalise_categories(record)
    _normalise_firm(record)

    losses = []
    prop = record["property"]
    for field, meta in PROPERTY_META.items():
        if meta.get("type") != "number":
            continue
        cell = prop.get(field)
        if cell is None:
            continue
        val = cell.get("value")
        if val is None or isinstance(val, (int, float)):
            continue

        rng = _number_range(val) if field in RANGE_NUMBER_FIELDS else None
        if rng is not None:
            lo, hi = rng
            lo_field, hi_field = RANGE_NUMBER_FIELDS[field]
            for name, num in ((lo_field, lo), (hi_field, hi)):
                target = prop.setdefault(name, {"value": None, "page": None,
                                                "snippet": None,
                                                "confidence": None})
                target["value"] = num
                if target.get("page") is None:
                    target["page"] = cell.get("page")
                if target.get("snippet") is None:
                    target["snippet"] = cell.get("snippet")
                if target.get("confidence") is None:
                    target["confidence"] = cell.get("confidence")
            cell["value"] = hi
            print(f"[coerce] property.{field}: {val!r} -> range {lo:g}-{hi:g} "
                  f"({field}={hi:g}, {lo_field}={lo:g}, {hi_field}={hi:g})",
                  file=sys.stderr)
            continue

        # "Not applicable" in a numeric property field is the report stating
        # the field does not apply - the same statement _NON_VALUES already
        # recognises on component rows. Nulling it is right; calling it a
        # coercion loss sent the whole report to review for saying so.
        if isinstance(val, str) and _NON_VALUES.match(val):
            cell["value"] = None
            print(f"[coerce] property.{field}: {val!r} -> null (stated as "
                  f"not applicable)", file=sys.stderr)
            continue

        n = _num(val)                      # recovers "15 stories" -> 15.0
        cell["value"] = n
        if n is None:
            losses.append({"where": "property", "field": field,
                           "kind": "type_coercion",
                           "detail": f"non-numeric value nulled: {val!r}"})
        else:
            extra = len(_all_numbers(val)) - 1
            note = f" ({extra} further number(s) in the source text ignored)" if extra > 0 else ""
            print(f"[coerce] property.{field}: {val!r} -> {n:g}{note}",
                  file=sys.stderr)

    losses += _coerce_rows(record)

    # Backfill the base field from the range fields.
    #
    # num_stories_min/max are in PROPERTY_FIELDS, so the extractor is now asked
    # for them directly - and when a property genuinely has a range it fills
    # them and leaves num_stories null, which means the branch above never sees
    # a string to parse and the base field stays empty. Anything keyed on
    # num_stories then silently loses those properties. The base field carries
    # the max, matching what the parse branch does.
    for field, (lo_field, hi_field) in RANGE_NUMBER_FIELDS.items():
        cell = prop.get(field)
        if cell is None or cell.get("value") is not None:
            continue
        hi = (prop.get(hi_field) or {}).get("value")
        lo = (prop.get(lo_field) or {}).get("value")
        if hi is None:
            continue
        cell["value"] = hi
        print(f"[coerce] property.{field}: null, backfilled from "
              f"{lo_field}/{hi_field} range {lo}-{hi} -> {hi}", file=sys.stderr)

    return losses


# ── layer 3: completeness ──────────────────────────────────────────────────

def completeness_checks(record: dict) -> list[dict]:
    """Catch silently-empty extractions.

    Reconciliation can't see these: when a sum has nothing to add up, the
    comparison is skipped and the report looks clean. An empty systems or
    components table is a failed extraction, not a valid result - unless the
    firm genuinely has no such table, which the extractor must then say by
    populating what the narrative provides.
    """
    issues = []
    prop = record["property"]

    if not record["systems"]:
        issues.append({"where": "systems", "field": "n_rows", "kind": "empty",
                       "detail": "no system rows extracted - either the report "
                                 "has no condition summary (derive from "
                                 "narrative) or extraction missed the table"})
    if not record["components"]:
        issues.append({"where": "components", "field": "n_rows", "kind": "empty",
                       "detail": "no component rows extracted"})

    # a stated total with no line items behind it is always wrong
    for total_field, table in [("immediate_repairs_total_usd", "immediate"),
                               ("short_term_repairs_total_usd", "short_term"),
                               ("non_critical_repairs_total_usd", "non_critical"),
                               ("priority_repairs_total_usd", "priority"),
                               ("operational_repairs_total_usd", "operational"),
                               ("reserves_total_uninflated_usd", "reserve")]:
        stated = _num((prop.get(total_field) or {}).get("value"))
        rows = [r for r in record["components"] if r.get("table") == table]
        if stated and stated > 0 and not rows:
            issues.append({"where": "components", "field": total_field,
                           "kind": "empty",
                           "detail": f"report states {table} total {stated} but "
                                     f"no {table} line items were extracted"})
        # rows present but every total null sums to nothing, which
        # reconciliation then skips as "can't compare" - same blind spot.
        if stated and stated > 0 and rows and \
                all(_num(r.get("total_cost_usd")) is None for r in rows):
            issues.append({"where": "components", "field": total_field,
                           "kind": "empty",
                           "detail": f"report states {table} total {stated} and "
                                     f"{len(rows)} {table} row(s) were extracted, "
                                     f"but none carries a total_cost_usd"})
    return issues


# ── layer 4: reconciliation ────────────────────────────────────────────────

def _close(a, b, rel=RECONCILE_REL_TOL):
    if a is None or b is None:
        return True   # can't compare -> not a reconciliation failure
    return abs(a - b) <= max(1.0, abs(a) * rel)


def _sum_or_none(values):
    """Sum, or None when there was nothing to add.

    Distinguishes "no rows contributed" (None, not comparable) from "the rows
    summed to zero" (0.0, comparable). `sum(...) or None` conflated them.
    """
    nums = [v for v in values if v is not None]
    return sum(nums) if nums else None


def reconciliation_checks(record: dict) -> list[dict]:
    issues = []
    prop = record["property"]
    p_imm = _num(prop["immediate_repairs_total_usd"].get("value"))
    p_res = _num(prop["reserves_total_uninflated_usd"].get("value"))

    s_imm = _sum_or_none(_num(r.get("immediate_repairs_usd"))
                         for r in record["systems"])
    s_res = _sum_or_none(_num(r.get("replacement_reserves_usd"))
                         for r in record["systems"])

    c_imm = _sum_or_none(_num(r.get("total_cost_usd"))
                         for r in record["components"]
                         if r.get("table") == "immediate")
    c_short = _sum_or_none(_num(r.get("total_cost_usd"))
                           for r in record["components"]
                           if r.get("table") == "short_term")
    p_short = _num(prop["short_term_repairs_total_usd"].get("value"))
    c_noncrit = _sum_or_none(_num(r.get("total_cost_usd"))
                             for r in record["components"]
                             if r.get("table") == "non_critical")
    p_noncrit = _num((prop.get("non_critical_repairs_total_usd") or {}).get("value"))
    c_prio = _sum_or_none(_num(r.get("total_cost_usd"))
                          for r in record["components"]
                          if r.get("table") == "priority")
    p_prio = _num((prop.get("priority_repairs_total_usd") or {}).get("value"))
    c_oper = _sum_or_none(_num(r.get("total_cost_usd"))
                          for r in record["components"]
                          if r.get("table") == "operational")
    p_oper = _num((prop.get("operational_repairs_total_usd") or {}).get("value"))
    c_res = _sum_or_none(_num(r.get("total_cost_usd"))
                         for r in record["components"]
                         if r.get("table") == "reserve")

    s_short = _sum_or_none(_num(r.get("short_term_repairs_usd"))
                           for r in record["systems"])
    s_noncrit = _sum_or_none(_num(r.get("non_critical_repairs_usd"))
                             for r in record["systems"])

    # Contingency markup, if this firm applies one. See RECONCILE_REL_TOL.
    c_pct = _num((prop.get("contingency_pct") or {}).get("value"))
    c_usd = _num((prop.get("contingency_usd") or {}).get("value"))

    def _reconciles(stated, summed):
        if _close(stated, summed):
            return True
        if stated is None or summed is None:
            return True
        if c_pct and _close(stated, summed * (1 + c_pct / 100.0)):
            return True
        if c_usd and _close(stated, summed + c_usd):
            return True
        return False

    # When a report states ONE combined near-term total but itemises into
    # several buckets (accessibility, deferred, critical...), comparing only
    # the rows tagged "immediate" under-sums and flags a correct extraction.
    # This is the same shape of fix as the contingency allowance: accept the
    # other legitimate reading, never a loose one.
    c_near = _sum_or_none(_num(r.get("total_cost_usd"))
                          for r in record["components"]
                          if r.get("table") in NEAR_TERM_TABLES)

    def _reconciles_immediate(stated, summed):
        return _reconciles(stated, summed) or _reconciles(stated, c_near)

    for name, stated, summed in [
        ("immediate: property vs systems", p_imm, s_imm),
        ("short_term: property vs systems", p_short, s_short),
        ("non_critical: property vs systems", p_noncrit, s_noncrit),
        ("immediate: property vs components", p_imm, c_imm),
        ("reserves: property vs systems", p_res, s_res),
        ("reserves: property vs components", p_res, c_res),
        ("short_term: property vs components", p_short, c_short),
        ("non_critical: property vs components", p_noncrit, c_noncrit),
        ("priority: property vs components", p_prio, c_prio),
        ("operational: property vs components", p_oper, c_oper),
    ]:
        ok = (_reconciles_immediate(stated, summed)
              if name.startswith("immediate") else _reconciles(stated, summed))
        if not ok:
            extra = ""
            if c_pct or c_usd:
                extra = (f" (contingency {c_pct or ''}{'%' if c_pct else ''}"
                         f"{c_usd or ''} applied and still short)")
            issues.append({"where": "cross-table", "field": name,
                           "kind": "reconcile",
                           "detail": f"stated {stated} != summed {summed}{extra}"})

    # each reserve row: year columns should sum to its total
    for i, row in enumerate(record["components"]):
        if row.get("table") != "reserve":
            continue
        # Gabion's schedule is escalated year by year, so it CANNOT equal an
        # uninflated total - a correct extraction of one of their reports
        # would fail this check on every single row. The extractor flags those
        # rows rather than trying to back the inflation out.
        if row.get("years_inflated"):
            continue
        years = [_num(row.get(f)) for f in YEAR_FIELDS]
        ysum = sum(v for v in years if v is not None)
        total = _num(row.get("total_cost_usd"))
        if total is not None and ysum > 0 and not _close(total, ysum):
            issues.append({"where": f"components[{i}]", "field": "total_cost_usd",
                           "kind": "reconcile",
                           "detail": f"year cols sum {ysum} != total {total}: "
                                     f"{row.get('description')!r}"})
    return issues


def arithmetic_checks(record: dict) -> list[dict]:
    """Row-internal arithmetic. Free, and independent of the model.

    Reconciliation proves the table sums to the report's stated totals; it
    cannot see a row whose own numbers contradict each other and which happens
    to sit inside a correct total. These are the identities the tables
    themselves assert:

        quantity x unit_cost  ==  cycle_replace (or total, when there is no
                                  separate cycle column)
        eul - effective_age   ==  rul

    Both are checked only when every term is present, and with the same
    tolerance reconciliation uses, because firms round displayed values. A
    failure here means a transcription error inside one row - a digit dropped
    from a unit cost, a quantity read off the wrong line.
    """
    issues = []
    for i, row in enumerate(record.get("components") or []):
        qty = _num(row.get("quantity"))
        unit_cost = _num(row.get("unit_cost_usd"))
        cycle = _num(row.get("cycle_replace_cost_usd"))
        total = _num(row.get("total_cost_usd"))
        extended = cycle if cycle is not None else total
        if (qty is not None and unit_cost is not None and extended is not None
                and qty > 0 and unit_cost > 0):
            expect = qty * unit_cost
            # An INTEGER multiple is recurrence, not an error. A component
            # replaced twice in the term costs 2 x quantity x unit cost, and
            # firms that express recurrence with a cycle length rather than a
            # replace-percent column (Gabion) have no other column to show it
            # in. The first version of this check flagged 121 rows, and every
            # sample inspected was an exact 2x - correct data, reported as a
            # fault. Only a NON-integer ratio indicates a real inconsistency.
            ratio = extended / expect if expect else 0
            near_int = abs(ratio - round(ratio)) <= 0.02 and 1 <= round(ratio) <= 20
            if (row.get("replace_percent") in (None, 100)
                    and row.get("qty_in_eval_period") is None
                    and not near_int
                    and not _close(expect, extended)):
                issues.append({
                    "where": f"components[{i}]", "field": "unit_cost_usd",
                    "kind": "arithmetic",
                    "detail": f"quantity {qty:g} x unit cost {unit_cost:g} = "
                              f"{expect:,.0f}, but the row states {extended:,.0f}"
                              f" ({row.get('description')!r})"})

        eul = _num(row.get("eul_years"))
        age = _num(row.get("effective_age_years"))
        rul = _num(row.get("rul_years"))
        if None not in (eul, age, rul) and abs((eul - age) - rul) > 1.5:
            issues.append({
                "where": f"components[{i}]", "field": "rul_years",
                "kind": "arithmetic",
                "detail": f"EUL {eul:g} - age {age:g} = {eul - age:g}, but RUL "
                          f"is {rul:g} ({row.get('description')!r})"})
    return issues


def number_presence_checks(record: dict, pdf_path: str) -> list[dict]:
    """Do the headline figures actually appear in the PDF?

    The strongest free check available on the property layer. Grounding tests
    the model's own snippet, which the model chose; this tests the VALUE, by
    searching the document text for the number as a human would see it
    written - with thousands separators, with and without decimals.

    Advisory, like grounding: a report can state a total only inside an image,
    and pdfplumber's text differs from what the model reads. A miss means
    "nobody could find this number in the text layer", which is worth knowing
    before the figure becomes a training label, not proof of an error.
    """
    money_fields = [f for f in PROPERTY_FIELDS
                    if f.endswith("_usd") and "per_unit" not in f]
    prop = record["property"]
    if not any(_num((prop.get(f) or {}).get("value")) for f in money_fields):
        return []

    hay = _squash(" ".join(_page_texts(pdf_path)))
    issues = []
    for f in money_fields:
        v = _num((prop.get(f) or {}).get("value"))
        if v is None or v <= 0:
            continue
        n = int(round(v))
        forms = {f"{n:,}", str(n), f"{n:,}.00", f"{v:,.2f}"}
        if not any(_squash(x) in hay for x in forms):
            issues.append({"where": "property", "field": f,
                           "kind": "not_in_text",
                           "detail": f"{v:,.0f} does not appear anywhere in the "
                                     f"PDF text layer"})
    return issues


# ── layer 5: grounding ─────────────────────────────────────────────────────

def _page_texts(pdf_path: str) -> list[str]:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            out.append(pg.extract_text() or "")
    return out


def grounding_checks(record: dict, pdf_path: str) -> list[dict]:
    """Property layer only: cited snippet must appear in the document.

    Matching is deliberately tolerant of HOW the text was extracted and strict
    about WHAT it says. The model and pdfplumber read the same PDF through
    different parsers, so requiring byte-identical agreement failed on text
    that was plainly present - which buries the real hallucinations in noise.
    """
    pages = _page_texts(pdf_path)
    norm_pages = [_norm(p) for p in pages]
    squash_pages = [_squash(p) for p in pages]
    # Built once. Previously rebuilt inside the loop for every property field.
    doc_norm = " ".join(norm_pages)
    doc_squash = "".join(squash_pages)

    fails = []
    for field in PROPERTY_FIELDS:
        cell = record["property"].get(field) or {}
        if cell.get("value") is None or not cell.get("snippet"):
            continue
        snippet = cell["snippet"]
        page = cell.get("page")
        # Reports carry their own printed page numbers (roman-numeral front
        # matter, section restarts), so the model's "page" often won't equal
        # the PDF page index. Treat page as a hint: check it first, then fall
        # back to the whole document. What actually guards against
        # hallucination is the snippet existing verbatim SOMEWHERE.
        if isinstance(page, int) and 1 <= page <= len(pages) and \
                _snippet_present(snippet, norm_pages[page - 1],
                                 squash_pages[page - 1]):
            continue
        if _snippet_present(snippet, doc_norm, doc_squash):
            continue   # present in the document, just not on the cited page
        fails.append({"where": "property", "field": field, "kind": "grounding",
                      "detail": f"snippet not found in document: {snippet!r}"})
    return fails


def collect_suspect_property_fields(record, det_issues, recon_issues, ground_fails):
    """Property fields worth an LLM-judge pass. Table problems route to
    needs_review directly — a judge can't repair a 60-row table reliably."""
    suspects = {i["field"] for i in det_issues if i["where"] == "property"}
    suspects |= {g["field"] for g in ground_fails}
    for field in PROPERTY_FIELDS:
        cell = record["property"].get(field) or {}
        conf = cell.get("confidence")
        if cell.get("value") is not None and isinstance(conf, (int, float)) \
                and conf <= CONFIDENCE_FLOOR:
            suspects.add(field)
    return sorted(suspects)