"""
Layer 0: extraction, tuned to the real PCR structure.

Changes from v1:
- Selects which pages of the PDF to send instead of blindly taking the first
  MAX_PDF_PAGES. These reports run 200+ pages and the extractable content
  (exec summary table, property description, Tables 1 & 2) normally sits in
  the front, with photo appendices behind it - but "normally" is not "always",
  and a report whose cost tables fall outside the slice produces empty
  component arrays that reconciliation cannot flag (a sum with no rows to add
  is skipped). So: keep a front block, then scan the text layer for cost-table
  markers and keep those pages too. Falls back to the old first-N behaviour if
  the scan finds nothing. The API caps PDF requests at 100 pages / 32MB.
- Page indices reported by the model refer to the SLICED pdf, so extract()
  maps them back to original 1-based page numbers before returning.
- Returns three layers: property (with per-field grounding), systems
  (exec summary table rows), components (Table 1 + Table 2 line items).
  Tables are validated by reconciliation instead of per-cell snippets.

Truncation handling (v3):
- Text is accumulated from the stream as it arrives, not read back off the
  final message. A text block cut off by max_tokens can lack its terminating
  event and be dropped from the accumulated message, which surfaces as
  "model returned no text" with the whole budget spent and nothing to show.
- A truncated component call is retried once, split into Table 1 and Table 2
  as separate calls, which halves the worst-case response size. Only reports
  that actually truncate pay the extra call.
- If a split call still truncates, complete row objects are salvaged from the
  partial JSON rather than throwing the whole report away. Salvage is loud on
  stderr and the missing rows make the table under-sum, so reconciliation
  routes the report to needs_review instead of letting it look clean.

Cross-firm work (v4), after reading all 132 unique reports in the inbox:
- Page selection no longer depends on knowing a firm's vocabulary. Marker
  matching is still there and has grown, but _structural_score adds the two
  signals that generalise - currency density and runs of consecutive calendar
  years used as column headers. Four firms (Gabion, LandScience, CBC, Tetra
  Tech) carried ordinary cost tables that scored ZERO on the old marker list.
- The blind first-N fallback now fires only when the text layer genuinely
  tells us nothing. It used to fire whenever every table page happened to sit
  inside the front block, which is the good case: ten reports were paying for
  a 90-page send to re-cover a front block already included in 30.
- Page selection is size-aware end to end. A slice over the request limit used
  to be sent anyway and rejected by the API; it now drops its least valuable
  pages until it fits, and says which cost-table pages it had to give up.
- The component prompt describes seven table LAYOUTS rather than one, because
  the corpus has seven - including Gabion's, which has no EUL, EFF AGE or RUL
  column at all and marks immediate vs reserve with a row class prefix.
- Reserve schedules run 5 to 15 years across firms, not 12 (see RESERVE_YEARS
  in schema.py), and some are quoted in calendar years and in inflated
  dollars. All three are now handled explicitly rather than silently dropped.
"""
import base64, hashlib, io, json, random, re, sys, time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
from langsmith import traceable
from pypdf import PdfReader, PdfWriter

from schema import (PROPERTY_FIELDS, SYSTEM_FIELDS, COMPONENT_FIELDS,
                    MAX_PDF_PAGES, RESERVE_YEARS, CONDITIONS)
from taxonomy import (SUBCATEGORIES, SUBCATEGORY_SCOPE, SUBCATEGORY_OTHER,
                      classify_subcategory, subcategory_for_component)

# The subcategory definition list the prompt shows the model, rendered from
# taxonomy.SUBCATEGORY_SCOPE so the instructions cannot drift from the schema.
# The old systems layer drifted into 1,216 distinct names precisely because
# the prompt described the target in prose that nothing validated against.
_SUBCAT_BLOCK = "\n".join(
    f"     {i:>2}. {name} - {SUBCATEGORY_SCOPE[name]}"
    for i, name in enumerate(SUBCATEGORIES, 1))

# Confirm the current model name in the Anthropic console; strings roll over.
MODEL = "claude-sonnet-5"

# Output ceiling per call. Raise if the console shows a higher cap for MODEL;
# the split-by-table retry below is the more reliable lever for long tables.
MAX_OUT = 32000

# Deliberation budget. This is not a cost knob - it is a correctness one.
#
# max_tokens is a ceiling on thinking AND answer together, and nothing
# reserves room for the answer. Baldwin Park proved what that costs: call A
# came back with stop_reason=max_tokens, thinking_tokens=32000 of 32000, and
# blocks=[('thinking', 0)] - the model deliberated through the entire budget
# and emitted no text at all, which surfaces as "model returned no text" after
# paying for a full extraction. Raising max_tokens alone does not fix this; it
# just makes the same failure more expensive.
#
# Bounding effort guarantees room for the answer. This is transcription of
# tables that are already structured, not open-ended reasoning, so the depth
# was never earning its cost either: measured thinking share was 32,000/32,000
# on that call A, 23,313/31,215 on a call B, and 7,796/8,000 in the judge.
# The retry below drops to "low" rather than raising the ceiling.
EFFORT = "medium"
RETRY_EFFORT = "low"

# Hard API limits per PDF request. MAX_PDF_PAGES is the budget we aim for;
# these are the walls. A slice over the size limit is rejected by the API, so
# the front block is trimmed until it fits rather than failing the report.
API_MAX_PAGES = 100

# The 32MB API limit applies to the REQUEST, and the PDF travels base64-encoded
# inside it - which is 4/3 the size of the bytes on disk. Checking the raw
# byte count against 30MB therefore passed a slice that arrived as a ~42MB
# request, and the API returned 413 request_too_large after the whole
# page-selection pass had run. 22MB of PDF encodes to ~29.3MB, leaving room
# for the instructions and JSON envelope.
#
# Found on 7800 Alpha Road: 93 pages whose text layer is a repeated watermark,
# so page selection cannot drop anything on score and falls back to trimming.
MAX_PDF_BYTES = 22 * 1024 * 1024
B64_OVERHEAD = 4 / 3
MIN_FRONT_PAGES = 10                    # never trim the exec summary away

# Pages always kept from the front: exec summary, "Project At a Glance",
# property description, and on most reports Tables 1 and 2 as well.
#
# MEASURED, do not lower without re-measuring. Mapping every grounded property
# value back to its original page across 134 extractions:
#
#     front block   reports with ALL property     reports with all KEY
#       size        fields inside the block       financial fields inside
#         20                17.9%                        84.3%
#         25                35.1%                        85.8%
#         30                53.0%                        88.8%
#         40                76.9%                        94.8%
#
# Even at 30, nearly half of all reports depend on pages the SCORER pulls in
# (cost-table pages, which often carry the stated totals) rather than on the
# front block alone. Cutting to 20 to save input tokens would put key
# financial fields at risk on about six more reports - a bad trade against
# roughly 15% of run cost. The cheap levers are the judge and the Batch API.
FRONT_PAGES = 30

# Phrases that mark a cost-table page. Matched case-insensitively; a page
# needs a score of 2 to count, so prose mentioning "unit cost" in passing does
# not qualify.
#
# This list is firm vocabulary, and firm vocabulary is exactly what a new firm
# does not share - see _structural_score below for the part that does not
# depend on knowing the words.
_TABLE_MARKERS = (
    # shared across most firms
    "eff age", "eul", "rul", "unit cost", "total cost",
    "immediate repair", "short term", "replacement reserve",
    "table 1", "table 2", "uninflated", "inflated",
    # shape A - EBI / Bureau Veritas / AEI / Nova / NV5 / Lender Consulting
    "cycle replace", "replace percent",
    # shape B - Partner Engineering
    "on site qty", "qty in eval",
    # shape C - Nova
    "base cost",
    # shape D - EMG
    "rul:eul", "rating 1-5",
    # Terracon and EPIC run reduced tables with no EFF AGE or RUL column, so
    # their cost pages carried only one of the markers above and would never
    # have been selected. These are the words those tables do use.
    "capital reserve", "reserve schedule", "reserve term",
    "total over term", "r-total", "eul/yrs",
    # shape E - Gabion. Their table has NO EUL/EFF AGE/RUL column at all, so
    # it scored 0 on everything above. These are its actual column headers.
    "capital considerations", "present worth", "work span",
    "of occurr", "start year",
    # shape G - the inline per-section cost summaries used by EBI, Atwell and
    # Metropolitan Solutions.
    "cost summary", "recommendation",
    # shape H - LandScience and CBC cost their work inside a section table
    # rather than a component table.
    "immed. cost", "reserve cost", "physical condition summary",
    "probable immediate repairs", "immediate need repair",
    # shape I - AEI's Homewood Suites set and Tetra Tech spell the life
    # columns out in words instead of abbreviating them.
    "expected life", "remaining life", "reflective age",
    "expected useful life", "remaining useful life", "capital items",
)

# Markers with every non-alphanumeric character removed. PDF text layers do
# not preserve spacing reliably: real reports in this corpus render "EFF AGE"
# as "EFFAGE", "Unit Cost" as "UnitCost", "Year 6" as "Year6". Matching the
# spaced form alone silently fails on those firms, and a missed table page is
# invisible - the report just comes back with fewer rows.
_SQUASHED_MARKERS = tuple(re.sub(r"[^a-z0-9]", "", m) for m in _TABLE_MARKERS)

# A currency amount, and a run of four or more consecutive calendar years.
_MONEY = re.compile(r"\$\s?[\d,]+")
_CALENDAR_RUN = re.compile(r"\b20\d\d\b(?:\s+\b20\d\d\b){3,}")

# Money amounts on a page before it is treated as a cost table on that basis
# alone. Tuned against the corpus: real cost-table pages carry 40-500 amounts,
# a narrative page that quotes a few repair costs carries under 10.
_MONEY_STRONG = 12
_MONEY_WEAK = 6


# max_retries above the SDK default of 2. A full-corpus run is ~4 hours of
# continuous calls and the account has hit its usage limit mid-run before
# (report 55 of 107). The SDK retries 429 and 5xx with exponential backoff, so
# a higher ceiling rides out a short throttle instead of failing a report that
# has already paid to upload its PDF.
_client = anthropic.Anthropic(max_retries=6)


class TruncatedOutput(ValueError):
    """Model output hit the token ceiling. Carries whatever text arrived."""

    def __init__(self, message: str, tag: str, raw: str, stop):
        super().__init__(message)
        self.tag, self.raw, self.stop = tag, raw, stop


def _page_text(page) -> str:
    try:
        return (page.extract_text() or "").lower()
    except Exception:
        return ""


def _marker_hits(text: str) -> int:
    """Distinct cost-table markers on a page, tolerant of lost spacing.

    Checked against the text as-is AND against a version with all punctuation
    and whitespace stripped, because PDF extraction mangles spacing
    inconsistently between firms ("EFFAGE", "UnitCost", "Year6"). A marker that
    only matches one form still counts once.
    """
    squashed = re.sub(r"[^a-z0-9]", "", text)
    return sum(1 for m, sm in zip(_TABLE_MARKERS, _SQUASHED_MARKERS)
               if m in text or sm in squashed)


def _structural_score(text: str) -> int:
    """What a cost table looks like when you do not know the firm's words.

    _marker_hits is a vocabulary list, and vocabulary is precisely what a firm
    we have never seen does not share. Four of the corpus's firms scored at or
    near zero on markers alone while carrying perfectly ordinary cost tables:

      Gabion       "Capital Considerations": Class / Item I.D. / Units /
                   Unit Cost / Present Worth / Start Year / Work Span, then
                   calendar-year columns. Not one EUL, EFF AGE or RUL.
      LandScience  costs itemised inside a section table under "Immed. Cost"
                   and "Reserve Cost".
      CBC          a numbered "Immediate Need Repairs Estimate" list.
      Tetra Tech   "EXPECTED LIFE / REFLECTIVE AGE / REMAINING LIFE".

    Their words are in _TABLE_MARKERS now, which fixes those four and nothing
    else. What generalises is the SHAPE: a page dense in currency amounts, or
    carrying a run of consecutive calendar years used as column headers, is a
    cost table whatever it calls its columns. A narrative page that quotes a
    couple of repair costs is not - hence the threshold rather than a flag.
    """
    score = 0
    money = len(_MONEY.findall(text))
    if money >= _MONEY_STRONG:
        score += 2
    elif money >= _MONEY_WEAK:
        score += 1
    if _CALENDAR_RUN.search(text):
        score += 2
    return score


def _score_pages(reader) -> list:
    """Cost-table score per page. Scored once; composition may run several
    times while fitting the slice under the size limit.

    extract_text() is by far the most expensive call in this module, so the
    page's text is pulled ONCE and everything is matched against the string.
    Calling _page_text(pg) inside the generator instead re-extracts the page
    for every marker - 16x the work, which on a 177-page report is 2,832 text
    extractions per slice rather than 177.
    """
    out = []
    for pg in reader.pages:
        text = _page_text(pg)
        out.append(_marker_hits(text) + _structural_score(text))
    return out


def _compose(scores: list, front_n: int, budget: int):
    """-> (front indices, table indices, table indices that did not fit)."""
    total = len(scores)
    front = list(range(min(front_n, budget, total)))
    candidates = sorted(((s, i) for i, s in enumerate(scores)
                         if s >= 2 and i >= len(front)),
                        key=lambda t: (-t[0], t[1]))
    room = max(0, budget - len(front))

    keep = set()
    for _, i in candidates:
        for j in (i - 1, i, i + 1):          # neighbours catch table spillover
            if 0 <= j < total and j >= len(front):
                keep.add(j)
        if len(keep) >= room:
            break

    tail = sorted(keep)[:room]
    missed = [i for _, i in candidates if i not in set(tail)]
    return front, tail, missed


def _encode(reader, idxs: list, pdf_path: str):
    if len(idxs) == len(reader.pages):
        raw = Path(pdf_path).read_bytes()
    else:
        writer = PdfWriter()
        for i in idxs:
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        raw = buf.getvalue()
    return base64.standard_b64encode(raw).decode(), len(raw)


@lru_cache(maxsize=8)
def _prepare_pdf(pdf_path: str, max_pages: int = MAX_PDF_PAGES):
    """-> (base64 pdf, page_map) where page_map[i] is the original 0-based
    index of the i-th page in the slice.

    Memoised, because this is called twice per report and is no longer cheap.
    judge.py re-slices the same PDF to send it a third time, and selecting
    pages means running extract_text() over every page of a 200-page document
    plus re-encoding and base64-ing the result. Doing that work once per
    report instead of twice is free latency.

    maxsize must exceed the worker count, not match it. At maxsize=3 with 3
    workers, the moment a fourth report starts it evicts the least-recently
    used entry - which can belong to a report that has finished extracting but
    has not reached its judge call yet, so that report re-scans and re-encodes
    its PDF from scratch. The tell is a duplicate [pages] line for the same
    report. 8 leaves room for the in-flight set plus churn; each entry holds
    one base64 slice, so this is bounded by page budget, not report count.
    Callers treat the page map as read-only.

    Selection, not truncation. A first-N slice silently drops whatever sits
    behind it, and missing table rows look like a report that simply had fewer
    line items - no layer of the validation stack can tell the difference.
    So: keep a front block, then keep the pages that actually carry cost-table
    markers wherever they are, and say on stderr when something had to be left
    out. Photo appendices are what gets cut, which is the point.
    """
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    budget = min(max_pages, API_MAX_PAGES)
    name = Path(pdf_path).name

    if total <= budget:
        data, size = _encode(reader, list(range(total)), pdf_path)
        if size <= MAX_PDF_BYTES:
            return data, tuple(range(total))
        # Short enough to send whole, too heavy to send whole. Fall through to
        # selection, which drops the pages that are not carrying tables -
        # photo appendices, which is where the megabytes are.
        print(f"[pages] {name}: {total}p / {size / 1e6:.1f}MB is under the "
              f"page budget but over the {MAX_PDF_BYTES / 1e6:.0f}MB request "
              f"limit; selecting pages instead of sending whole",
              file=sys.stderr)

    scores = _score_pages(reader)

    # Did the text layer tell us anything at all? Previously the fallback to a
    # blind first-N slice also fired whenever every table page happened to sit
    # INSIDE the front block - which is the good case, not a failure - and ten
    # reports in this corpus paid for 90 pages on every call to re-send a
    # front block they had already covered in 30. The fallback belongs only
    # where the scan genuinely found nothing: a scanned report, or one whose
    # text layer is a watermark (7800 Alpha Road: 93 pages, zero currency
    # amounts, every page reading "Confidentially provided to ...").
    scored_any = any(s >= 2 for s in scores)
    if not scored_any:
        print(f"[pages] WARNING {name}: no page scored as a cost table - the "
              f"text layer is missing or unusable. Falling back to the first "
              f"{min(budget, total)} pages; expect thin extraction and no "
              f"grounding.", file=sys.stderr)

    front_n = FRONT_PAGES
    dropped: set = set()

    while True:
        front, tail, missed = _compose(scores, front_n, budget)
        if scored_any:
            idxs = sorted(front + tail)
        else:
            idxs = list(range(min(budget, total)))
        idxs = [i for i in idxs if i not in dropped]

        data, size = _encode(reader, idxs, pdf_path)
        if size <= MAX_PDF_BYTES:
            break

        # Trimming the front block only helps when the slice is actually
        # composed of front + table pages. In the fallback case the slice IS
        # the front block, so shrinking front_n changes nothing and the loop
        # would just log four no-op trims before getting to the real lever.
        if scored_any and front_n > MIN_FRONT_PAGES:
            front_n = max(MIN_FRONT_PAGES, front_n - 5)
            print(f"[pages] {name}: slice was {size / 1e6:.1f}MB, trimming "
                  f"front block to {front_n} pages", file=sys.stderr)
            continue

        # Front block is at its floor and the slice is still too heavy. Drop
        # the least valuable page and try again, rather than sending a request
        # the API will reject outright. Least valuable = lowest table score,
        # and among equals the latest page, since the exec summary is at the
        # front. The first MIN_FRONT_PAGES are never dropped.
        droppable = [i for i in idxs if i >= MIN_FRONT_PAGES]
        if not droppable:
            print(f"[pages] WARNING {name}: {size / 1e6:.1f}MB with only the "
                  f"first {MIN_FRONT_PAGES} pages left; the API will reject "
                  f"this request. The PDF needs downsampling before it can be "
                  f"processed.", file=sys.stderr)
            break
        dropped.add(min(droppable, key=lambda i: (scores[i], -i)))

    if tail:
        print(f"[pages] {total}p report: kept front 1-{len(front)} plus "
              f"{len(tail)} table page(s) at {[i + 1 for i in tail][:12]}"
              f"{' ...' if len(tail) > 12 else ''}", file=sys.stderr)
    if missed:
        # The one case that must never be quiet: real table pages were found
        # and could not be sent. Their rows will be missing, the table will
        # under-sum, and reconciliation should flag the report - but the cause
        # is the page budget, not the model, and only this line says so.
        print(f"[pages] WARNING {total}p report: {len(missed)} page(s) with "
              f"cost-table markers did not fit the {budget}-page budget "
              f"(pages {[i + 1 for i in missed][:12]}). Raise MAX_PDF_PAGES "
              f"in schema.py or lower FRONT_PAGES.", file=sys.stderr)
    lost = sorted(i for i in dropped if scores[i] >= 2)
    if lost:
        # Same failure as `missed`, different cause: these pages fitted the
        # page budget and were thrown out to get under the SIZE limit. Rows
        # from them will be missing.
        print(f"[pages] WARNING {name}: dropped {len(lost)} page(s) that "
              f"scored as cost tables to fit the size limit "
              f"(pages {[i + 1 for i in lost][:12]}). Reconciliation should "
              f"flag this report.", file=sys.stderr)

    return data, tuple(idxs)


def _sliced_pdf_b64(pdf_path: str, max_pages: int = MAX_PDF_PAGES) -> str:
    """Back-compat shim for callers that only want the encoded bytes."""
    return _prepare_pdf(pdf_path, max_pages)[0]


def pdf_block(data: str, cache: bool = True) -> dict:
    """The document content block, marked as a prompt-cache breakpoint.

    The same PDF is sent three times per report - extract call A, extract call
    B, and the judge - at full input price each time, and the PDF is nearly all
    of the input. Marking it cached makes sends two and three cache reads at
    roughly a tenth of input cost, which is the largest single saving available
    on a 135-report run and cuts latency too, since there is less to process.

    The block must be byte-identical across calls for the cache to hit, which
    is why the varying instructions go AFTER it in the content list and why
    every caller builds the block here rather than inline.

    Cache entries live 5 minutes by default. If LangSmith shows
    cache_read_input_tokens at zero on the B call, the gap between calls is
    exceeding that - pass {"type": "ephemeral", "ttl": "1h"} instead, which
    costs more to write but cannot miss on a slow report.
    """
    block = {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf",
                        "data": data}}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


_COMMON_RULES = """
Rules for everything:
- Never invent values. Numbers as numbers (no "$"/commas). Dates YYYY-MM-DD.
- Column names vary by firm, and the SAME fact is spelled many ways. Map any
  of these onto our names:
    eul_years            "EUL", "Avg EUL (YR)", "EUL/Yrs", "Expected Life",
                         "Overall Expected Life", "Expected Useful Life",
                         "Estimated Useful Life", "Life Expectancy"
    effective_age_years  "EFF AGE", "Eff Age (YR)", "Effective Age",
                         "Reflective Age", "Present Age", "Apparent Age"
    rul_years            "RUL", "RUL (YR)", "Remaining Life", "ERUL",
                         "Remaining Useful Life", "Est. Remaining Life"
    unit_cost_usd        "Unit Cost", "$ Cost", "Cost", "Unit Price"
    cycle_replace_cost_usd  "Cycle Replacement", "Cycle Replace",
                         "Present Worth", "Extended Cost", "R-Total$"
    total_cost_usd       "Total Cost", "Total", "Total Reserves Over Term",
                         "Total Over Term", "5 Year Totals", "Costs",
                         "Cumulative"
    quantity / unit      "Quantity"/"Qty"/"Number of Units" and
                         "Unit"/"Units"/"UOM". Keep the unit verbatim but
                         strip trailing periods: "Sq. Ft." -> "SF",
                         "Ea." -> "EA", "Lump sum" -> "LS", "Allow." -> "ALW".
- A life or age column may hold a RANGE ("5-7", "10-15", "20-25") rather than
  one number - AEI and Terracon both do this. Report the range as a string
  exactly as printed ("10-15"); it is split into min/max downstream. Do NOT
  average it, and do NOT pick one end.
- "Varies" / "var" in a life or age column is not a number: leave the numeric
  field null.
"""

# Call 1: property + systems. Snippets only here, where grounding matters.
_INSTRUCTIONS_A = f"""You are extracting structured data from a Property
Condition Assessment / Property Condition Report (ASTM E 2018 style).

Return ONLY a JSON object (no prose, no markdown fences) with exactly two keys:
"property" and "systems".

1. "property": an object with exactly these keys:
{json.dumps(PROPERTY_FIELDS)}
   Each key maps to: {{"value": <value or null>, "page": <1-based PDF page
   index counting the very first page of the file as 1 - NOT the page number
   printed on the page, which often differs - or null>,
   "snippet": <verbatim supporting phrase, max 12 words, or null>,
   "confidence": <0.0-1.0>}}
   - Keep snippets SHORT: the shortest verbatim phrase that supports the value.
   - report_firm is the ASSESSING firm - the engineering consultancy that
     inspected the property and WROTE this report. It is the name after
     "Prepared by", the logo on the cover, and almost always the name stamped
     in the page header or footer of EVERY page along with the project number.
     client_name is who COMMISSIONED it - the name after "Prepared for", "For",
     or "Confidentially provided to": an owner, lender, or investor.
     These are easy to swap and swapping them is silent - both are plausible
     company names and nothing downstream can tell. It has already happened:
     an AEI report came back as "Bridge House Advisors Corp." (the lender) and
     an EPIC report as "Partners". If two company names appear on the cover,
     the one repeated in the running header/footer of the body pages is the
     assessing firm; the one appearing only once on the cover is the client.
   - renovation_years / facade_materials / roof_types: join multiples with "; ".
   - fire_sprinklers / emergency_generator / basement: describe briefly if the
     report addresses them. Use "none" ONLY when the report affirmatively says
     the feature is absent. If the report simply never mentions it, use null.
     "none" is a finding; null is missing data - do not conflate them, and
     never infer absence from silence.
   - overall_condition: the report's own word, lowercased. Prefer
     excellent/good/fair/poor. If the firm's scale uses a different word
     ("average", "satisfactory", "adequate", "acceptable", "serviceable",
     "functional", "marginal"), USE THAT WORD - do not translate it onto
     good/fair. Firms disagree about where those sit: one report's legend
     makes "Average" a level between Good and Fair, another defines "Good" AS
     "average to above-average condition". Recording the word the report used
     keeps that distinction; translating it destroys it. "New" or "like new"
     may be reported as "excellent" - most legends in this corpus define
     Excellent as exactly that.
   - The near-term budget is split differently by different firms, and each
     split has its own stated total. Fill only the ones the report states:
       * short_term_repairs_total_usd - firms that split by TIMING (EMG,
         Bureau Veritas: an Immediate column and a Short Term column).
       * priority_repairs_total_usd / operational_repairs_total_usd - reports
         that tier near-term work by URGENCY ("Priority Repairs $53,000",
         "Operational Repairs $24,700").
       * non_critical_repairs_total_usd - firms that split by SEVERITY. Tetra
         Tech reports "Immediate Critical Repair Needs" and "Immediate
         Non-critical Repair Needs" as separate numbered sections with
         separate totals: the critical total goes in
         immediate_repairs_total_usd, the non-critical total here. Do NOT put
         a non-critical total in short_term_repairs_total_usd - severity and
         timing are different facts.
   - contingency_pct / contingency_usd: many firms add a contingency or fee ON
     TOP of the line items, so the items will not sum to the stated total
     without it. Gabion prints it in the table footer as
     "Contingency: 10.0%" with its own dollar row. Record both the percentage
     and the dollar amount when stated; leave null if the report has none.
     Do NOT fold the contingency into the individual line items.
   - reserves_total_uninflated_usd must be the sum of the RESERVE line items
     only. If the report's only headline figure combines reserves with
     immediate needs ("Total Expenditures (Including Immediate Needs)"), leave
     this null rather than using the combined number.
   - reserves_total_present_value_usd: only for a DISCOUNTED figure, e.g.
     "Total Present Value (With Contingency)". A present value is not a sum of
     line items and must never be placed in the uninflated or inflated total.
   - The DENOMINATOR differs by asset class, and getting it wrong silently
     corrupts every per-unit cost metric:
       * Apartments / multifamily -> num_units.
       * Senior housing (assisted living / memory care / independent living)
         is measured in BEDS: set num_beds, and leave num_units null unless
         the report states units separately. care_types = semicolon-joined
         care levels.
       * Hotels are measured in ROOMS or KEYS: set num_rooms. A report saying
         "# of Rooms: 108" or "Cost/Year/Room" or "Reserve per Key" is a
         hotel; do not put that count in num_units.
     unit_basis = which of these the report's own per-unit figures are per:
     "units", "beds", "rooms", or "sf". Set it whenever the report states a
     per-unit reserve or per-unit repair figure.
   - building_age_years: only if the report states it directly.

2. "systems": an array of EXACTLY 12 objects, in exactly this order, one per
   subcategory. Each object has exactly these keys:
{json.dumps(SYSTEM_FIELDS)}
   (property_id and report_firm may be null - they are filled in later.)

   THE 12 SUBCATEGORIES AND WHAT EACH COVERS:
{_SUBCAT_BLOCK}

   - RETURN ALL 12, ALWAYS, EVEN IF THE REPORT IS SILENT ON SOME. This array
     is a fixed-width row block, not a list of findings. For a subcategory the
     report does not address at all: "assessed": false, condition null, all
     costs null, notes null. For one it does address: "assessed": true.
     "Not assessed" and "assessed and found fine" are different facts and the
     `assessed` flag is the only thing that separates them - never drop a row
     to mean absence and never invent a condition to fill one.
   - `subcategory` is exactly the slug above. No other value is accepted.
   - FOLD the report's own sections into these twelve. A firm's "3.4 Roofing",
     "3.5 Roof Drainage" and "Roof Coverings" are all `roofing`. Record which
     of the firm's sections you folded in - their numbers and headings, joined
     with "; " - in `source_sections` ("3.4 Roofing; 3.5 Roof Drainage"). That
     field is the audit trail back to the page; a reviewer who cannot retrace
     a number cannot check it, so do not leave it null on an assessed row.
   - WHERE THE BOUNDARIES ACTUALLY FALL. These are the folds that get made
     wrongly, and the specification settles each one:
       * Site lighting and exterior/parking-lot lighting -> site_improvements,
         NOT electrical. Electrical is the building's service, distribution,
         panels, wiring and the emergency generator.
       * Roof drainage, gutters and downspouts -> roofing. Storm water and
         site/surface drainage -> site_improvements. Both are "drainage".
       * Retaining walls, perimeter walls, fencing and signage ->
         site_improvements, NOT structural or envelope.
       * Stairs, balconies and load-bearing walls -> structural_frame_foundation.
         Exterior walls, cladding, windows, doors, sealants and waterproofing
         -> building_envelope.
       * The emergency generator -> electrical. For SENIOR HOUSING the
         specification also names generators under additional_considerations
         alongside nurse call and commercial kitchen/dining: put the generator
         in electrical and note any senior-specific dependency in the
         additional_considerations notes. Do not put its cost in both.
       * Pools, laundry, commercial kitchen equipment, dining infrastructure
         and nurse call -> additional_considerations. Interior common-area and
         unit finishes, flooring, ceilings, cabinetry and appliances ->
         interior_elements.
       * A "Utilities" or "Utility Providers" section is usually about who
         supplies the site. Distribute any actual findings into plumbing
         (water, sewer, gas) or electrical (power service); if it is purely a
         list of providers with no condition or cost, it belongs to no
         subcategory - leave it out rather than forcing it into one.
   - EVERY DOLLAR LANDS IN EXACTLY ONE SUBCATEGORY. The twelve rows are summed
     and reconciled against the report's stated immediate and reserve totals,
     so a cost counted in two subcategories breaks the tie-out just as badly
     as one that is missed. When a section spans two subcategories, put the
     cost where the WORK is, not where the section heading sits.
   - Split each row's costs the SAME WAY as the cost tables, into
     immediate_repairs_usd / short_term_repairs_usd /
     non_critical_repairs_usd / replacement_reserves_usd. Do not fold a
     short-term or non-critical figure into immediate_repairs_usd: the
     systems sum is reconciled against the report's stated immediate total,
     and folding makes it overshoot.
   - Blank cost cells are null, not 0. A subcategory the report assessed and
     assigned no cost to has assessed=true and null costs - not 0.
   - replacement_reserves_usd is this subcategory's total over the WHOLE term,
     and this is the single most common systems-layer error. If the report
     shows a per-occurrence cost for something that RECURS, add up the
     occurrences. Verbatim example (Tetra Tech): "Asphalt Pavement
     Seal/Stripe ... $17,920" with $17,920 landing in BOTH year 2 and year 7 -
     the correct figure is 35,840, not the 17,920 printed in the row. Same
     rule as total_cost_usd on the cost tables: when a year schedule exists,
     the year cells are authoritative over any printed per-row total.
   - condition: same rule as overall_condition above - the report's own word,
     lowercased, not translated onto a four-point scale. When the folded
     sections disagree (roofing "good", roof drainage "fair"), condition = the
     BEST rating and condition_secondary = the rest joined with "; ", which is
     the same convention as a two-X rating below.
   - Ratings marked with two X's: condition = the better rating,
     condition_secondary = the worse. One X: condition_secondary = null.
     Three or more marked: condition = the best, condition_secondary = the
     rest joined with "; ".
   - NUMERIC RATINGS: some firms (EMG) rate 1-5 instead of using words. Put
     the number in condition_rating_numeric and leave condition null unless
     the report ALSO prints a word for that row. Do NOT convert the number to
     a word from memory - scales differ in direction between firms, and a
     report that prints no legend does not tell you which way its scale runs.
     If the report DOES print a legend ("1 - Excellent, 2 - Good ..."), you
     may fill both the number and the word it maps to. When folded sections
     carry different numbers, use the WORST (highest-wear) number and say so
     in notes.
   - rul_years: the remaining useful life the report states for this
     subcategory as a whole, most often printed for roofing. Null unless the
     report states it - do NOT compute it from age and expected life, and do
     not carry one component's RUL up to the whole subcategory. A negative
     value is valid and means past its expected life; keep the sign.
   - action_required: the report's own words, condensed ("Replace",
     "Refurbish, Repair", "None"). When folding several sections, join the
     distinct actions with "; ".
   - notes: at most 25 words of the report's own findings for this
     subcategory - what it is, what is wrong with it. Null when not assessed.
     This is the only narrative that survives the fold, so spend it on the
     finding, not on restating the subcategory name.
   - SOURCE: prefer the EXECUTIVE SUMMARY / "Project At a Glance" / "Physical
     Condition Summary" / "General Condition Description" table if the report
     has one. SOME FIRMS HAVE NO SUCH TABLE - Partner Engineering states
     conditions only in narrative ("The split systems appeared to be in good
     condition"). In that case fill the twelve rows from the prose. NEVER
     return fewer than 12 objects, and never return a report's systems as an
     empty array.
   - Some firms (LandScience, CBC) cost their work AT THIS LEVEL rather than
     in a component table: the section table carries "Immed. Cost" and
     "Reserve Cost" columns, or an "Action" cell holding numbered items with
     a price each. Fill immediate_repairs_usd / replacement_reserves_usd from
     those columns, and put the action text in action_required.
{_COMMON_RULES}"""

# Call 2: components only. No snippets - these are validated by reconciliation
# against the report's own stated totals, which is a stronger check anyway.
_INSTRUCTIONS_B = f"""You are extracting the COST TABLES from a Property
Condition Assessment report: TABLE 1 (immediate / short-term repairs) and
TABLE 2 (replacement reserves).

Return ONLY a JSON object (no prose, no markdown fences, NO pretty-printing -
emit it COMPACT on as few lines as possible) with exactly one key:
"components" - an array, one object per line item.

OUTPUT SIZE RULES - these matter, long tables get truncated otherwise:
- OMIT any key whose value would be null. Do not emit "eul_years": null.
  Missing keys are filled in as null automatically.
- Do NOT emit year_1..year_{RESERVE_YEARS}. Instead emit a single "years" object holding
  ONLY the years with a non-zero amount, keyed by TERM year number (1 = first
  year of the term), never by calendar year:
      "years": {{"3": 756, "8": 756}}
  Omit "years" entirely for immediate/short_term rows, and omit zero cells.
  Terms run 5 to {RESERVE_YEARS} years depending on the firm; use as many as the
  table actually has.
- Use compact JSON: no newlines between keys, no indentation.

Per-row keys (omit any that are null):
  table, section_code, description, eul_years, effective_age_years,
  rul_years, rul_varies, quantity, unit, qty_in_eval_period, unit_cost_usd,
  cycle_replace_cost_usd, replace_percent, start_year, cycle_years,
  total_cost_usd, years_inflated, years

- CRITICAL - what total_cost_usd MEANS. It is the total this line item costs
  OVER THE WHOLE EVALUATION TERM. Firms print a column headed "Total Cost"
  that sometimes means that and sometimes does not, so do not copy the column
  by its name - check it:
    * If the row has a year schedule, total_cost_usd MUST equal the sum of
      that row's year cells. That is the number reconciliation checks.
    * If the printed "Total Cost" is SMALLER than the year sum, it is a
      per-occurrence (one cycle) figure for an item that recurs. Put the
      printed figure in cycle_replace_cost_usd and set total_cost_usd to the
      year sum. Real example (Tetra Tech): "Asphalt Pavement Seal/Stripe ...
      TOTAL COST $17,920" with $17,920 falling in BOTH year 2 and year 7 - the
      row costs $35,840 over the term, and $17,920 is one seal/stripe.
    * If the layout has NO total column at all, derive total_cost_usd from the
      year cells. Never leave it null on a row that has a schedule: a table of
      rows with no totals cannot be reconciled against anything, and the whole
      report gets rejected.
- IMPORTANT - immediate vs short term: some firms (e.g. EMG, Bureau Veritas)
  split Table 1 into an "Immediate" column and a "Short Term" column and report
  their totals separately. Set table="immediate" ONLY for costs in the
  immediate column, and table="short_term" for the short-term column.
  total_cost_usd = that row's cost in ITS column. Never add the two together.
- OTHER NEAR-TERM BUCKETS. Firms slice near-term work in still more ways, and
  each gets its own table value rather than being forced into "immediate":
  table="critical" (critical/life-safety repairs listed separately),
  table="deferred" (deferred maintenance as its own category), and
  table="accessibility" (ADA/accessibility work costed separately). Use the
  report's own grouping - the totals are reconciled per bucket AND against
  their combined sum, so an honest grouping always ties out.
- URGENCY TIERS (Freddie-Mac-style reports, e.g. Villa Oaks): some reports
  split near-term work into "Immediate", "Priority Repairs" and "Operational
  Repairs", each with its own stated total, before a separate "Capital Needs
  Over the Loan Term" section. Use table="priority" and table="operational"
  for those two; capital needs are table="reserve". Keep them separate - each
  total is reconciled on its own.
- SEVERITY SPLIT (Tetra Tech): this firm divides costs three ways, not two -
  "Immediate Critical Repair Needs" (health and safety), "Immediate
  Non-critical Repair Needs", and replacement reserves, each in its own
  numbered section with its own total. Use table="immediate" for critical,
  table="non_critical" for non-critical, table="reserve" for the rest. Do not
  merge critical and non-critical, and do not call non-critical "short_term" -
  both are immediate in timing; they differ in severity.
- Table 2 rows: table="reserve"; put the spend schedule in the sparse "years"
  object described above.
- Table 1 / short-term rows: omit eul/age/rul and omit "years".
- EUL/EFF AGE/RUL of "var"/"Varies": numeric field null, rul_varies true.
  Otherwise rul_varies false.
- Some firms (e.g. Partner) have NO replace-percent column and instead give
  "On Site Qty" plus "Qty in Eval Period": quantity = on-site qty,
  qty_in_eval_period = eval-period qty, replace_percent = null. Firms with a
  Replace Percent column: record it as a number (60, 100, 300, 1200) and set
  qty_in_eval_period null. replace_percent may be BELOW 100 (partial).
- CALENDAR-YEAR COLUMN HEADERS. Several firms head the schedule with calendar
  years (2023 2024 2025 ...) instead of Year 1, Year 2. Convert: the earliest
  column is term year 1. Do not emit calendar years as keys.
- INFLATED SCHEDULES. Most firms print uninflated year columns and give the
  inflated figure on a separate total line. A few (notably the "Capital
  Considerations" layout below) print the year columns already escalated - the
  same $51,500 item shows as $65,239 in a later year. When the year columns
  are inflated, set years_inflated true on those rows. Say so rather than
  trying to back out the inflation.
- Do NOT include "Totals", "Total (Uninflated)", "Total (Inflated)",
  "Inflation Factor" or "Cumulative" rows as line items. They are the check
  we reconcile against, not data.
- NEVER EMIT AN ITEM TWICE, once uninflated and once inflated. Most reports
  present the same schedule in both forms - an uninflated table and an
  inflated one, or paired "un-inflated"/"inflated" totals. They are two views
  of THE SAME line items, not two sets of work. Emit each item ONCE, using the
  UNINFLATED figures, and let the inflated headline number go to the property
  field reserves_total_inflated_usd. This has already gone wrong: one report
  stating $404,682 uninflated and $464,129 inflated came back with components
  summing to $868,811 - exactly the two added together.
- Likewise do not emit both a summary row and the detail rows that make it up.
  If a group total is followed by its constituent items, emit the items only.
- Include EVERY line item: the totals will be checked against the report's
  own stated totals, so a missing row will be detected.

TABLE LAYOUTS YOU WILL MEET. Identify which one this report uses, then map it:

  * Replace-percent layout (EBI, Bureau Veritas, AEI, Nova, NV5, Lender
    Consulting): EUL / EFF AGE / RUL / Quantity / Unit / Unit Cost / Cycle
    Replace / Replace Percent / Year 1..N / Total Cost. EPIC uses the same
    layout minus the Cycle Replace and Replace Percent columns.
  * Qty-in-eval layout (Partner Engineering): no Replace Percent column;
    "On Site Qty" -> quantity, "Qty in Eval Period" -> qty_in_eval_period.
  * "CAPITAL CONSIDERATIONS" layout (Gabion). This one shares almost nothing
    with the others - read it carefully:
      - There is NO EUL, EFF AGE or RUL column. Leave all three null. Do not
        derive them from anything else.
      - Columns are: Class / Item I.D. / Item / Units / Number of Units /
        Unit Cost / Present Worth / Start Year of Occurrence / Work Span
        Cycle / then calendar-year columns.
      - "Number of Units" -> quantity, "Units" -> unit,
        "Present Worth" -> cycle_replace_cost_usd,
        "Start Year of Occurrence" -> start_year,
        "Work Span" / "Cycle" -> cycle_years.
      - The row's CLASS decides the table: a row marked "I.N." (immediate
        need) is table="immediate"; a row marked "R.R." (replacement reserve)
        is table="reserve". There is no separate Table 1 and Table 2.
      - Its year columns are inflated: set years_inflated true.
      - There is NO total column. Set total_cost_usd to the sum of that row's
        calendar-year cells - that is what their own "Total Expenditures"
        figure adds up. Present Worth is the undiscounted extended cost and
        belongs in cycle_replace_cost_usd, not here.
  * R-numbered reserve schedule (Terracon): Item Description / EUL / Quantity
    / Units / Cost / R-Total$ / Year 1..10 / Cumulative. No EFF AGE, no RUL -
    leave them null. EUL is usually a range ("5-7"). "R-Total$" ->
    total_cost_usd. Ignore the Cumulative column.
  * Inline per-section cost summaries (EBI, Atwell, Metropolitan Solutions):
    no single consolidated table. Each narrative section ends in a small block
    "COST SUMMARY / Recommendation / EUL / EFF AGE / RUL / Year / Cost", and
    the Year/Cost pair repeats for each occurrence. Emit ONE ROW PER
    RECOMMENDATION, put the occurrences in "years", and set section_code from
    the section the block sits in. A Year value of "Immed" means
    table="immediate", not term year 1.
  * Narrative-costed reports (LandScience, CBC, JLL/Merritt & Harris, Sierra
    Piedmont): there is no component table at all. Costs appear as numbered
    items inside a section-level table with "Immed. Cost" and "Reserve Cost"
    columns, as a numbered "Immediate Need Repairs Estimate" list, or as prose
    recommendations each ending in a price - "JLL recommends the fireproofing
    on the cellar steel be evaluated ... Estimated Cost $9,000". Build one
    component row per costed item: description from the item text,
    total_cost_usd from its price, table from the heading it sits under
    ("IMMEDIATE", "Priority", "Capital Needs"), section_code from its section.
    Leave the life columns null - the report does not state them.
    SWEEP THE WHOLE NARRATIVE. These items are scattered across the body of
    the report, not gathered in one place, and the stated totals are checked
    against your sum - a report whose narrative items sum to half its stated
    immediate total has been half-read. Do NOT return an empty or near-empty
    components array just because there is no formal cost table.
  * Costs quoted as a LOW-TO-HIGH RANGE (Sierra Piedmont: "a low to high range
    is given"): use the figure the report's own stated total is built from -
    normally the high end. Put that in total_cost_usd. Never average silently.
  * Spelled-out life columns (AEI's Homewood Suites reports, Tetra Tech):
    "Overall Expected Life" / "Remaining Life", or "EXPECTED LIFE" /
    "REFLECTIVE AGE" / "REMAINING LIFE". Same meanings as EUL / EFF AGE / RUL.
    AEI quotes both as ranges ("10-15", "1-2"); pass them through as strings.
{_COMMON_RULES}"""

# Scope suffixes for the split retry. Same rules, half the rows per response.
_SCOPE_T1 = """

SCOPE FOR THIS CALL: extract ONLY Table 1 rows - the immediate and short-term
repair line items (table="immediate" and table="short_term"). Do NOT emit any
replacement-reserve rows in this call."""

_SCOPE_T2 = """

SCOPE FOR THIS CALL: extract ONLY Table 2 rows - the replacement-reserve line
items (table="reserve"). Do NOT emit any immediate or short-term rows in this
call."""


# ── prompt versioning ──────────────────────────────────────────────────────
# A fingerprint of everything that decides what an extraction CONTAINS: both
# instruction strings, the model, and the three field lists. Any edit to any of
# them changes this hash.
#
# WHY IT EXISTS. The single worst problem in this corpus is documented in the
# handoff: the 134 extractions were produced under roughly six different prompt
# versions over one session, so extraction quality correlates with WHEN a report
# happened to run - and because reports ran in firm-grouped batches, it
# correlates with FIRM, which is the leave-one-firm-out CV axis. Nothing in the
# cached records recorded which prompt produced them, so there was no way to
# tell a stale extraction from a current one except by re-running everything.
#
# With the stamp, `batch.py` reuses a cached extraction only if it was produced
# by the CURRENT prompt and re-extracts it otherwise. Two things follow:
#   * A full re-extraction is RESUMABLE. The previous full run hit the account
#     usage limit at report 55 of 107; restarting with --no-cache would have
#     re-paid for all 55. Now the same command picks up where it stopped,
#     because the finished reports carry a matching stamp.
#   * "Extracted under one frozen prompt" becomes a checkable property of the
#     dataset instead of a claim - see `batch.py --prompt-audit`.
PROMPT_VERSION = hashlib.sha256("|".join([
    _INSTRUCTIONS_A, _INSTRUCTIONS_B, MODEL,
    ",".join(PROPERTY_FIELDS), ",".join(SYSTEM_FIELDS),
    ",".join(COMPONENT_FIELDS),
]).encode()).hexdigest()[:12]


def _salvage_rows(raw: str) -> list:
    """Complete row objects recoverable from a truncated components array.

    Missing rows make the table under-sum against the report's stated totals,
    so reconciliation flags the report. This recovers most of a long table
    instead of discarding the whole report; it never hides the shortfall.
    """
    i = raw.find('"components"')
    i = raw.find("[", i) if i != -1 else raw.find("[")
    if i == -1:
        return []

    rows, depth, start, in_str, esc = [], 0, None, False, False
    for j in range(i + 1, len(raw)):
        ch = raw[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = j
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    rows.append(json.loads(raw[start:j + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
        elif ch == "]" and depth == 0:
            break
    return rows


# ── transient-failure retry ────────────────────────────────────────────────
# The SDK's `max_retries` covers the INITIAL HTTP request only. An
# `overloaded_error` can also arrive as an SSE event AFTER the stream has
# opened, and the SDK raises it straight out of `for chunk in
# stream.text_stream` with no retry - which is exactly how Brookdale Irving
# died 73 reports into a 132-report run, having already paid to upload its PDF.
#
# So the retry has to wrap the whole stream, not the request. Re-running a
# stream from scratch is safe: the call is idempotent, and the document block
# is cache-marked, so a second attempt within the TTL re-reads the cache at a
# tenth of input cost rather than re-uploading.
STREAM_RETRIES = 4
STREAM_BACKOFF = 8.0     # seconds, doubled each attempt, plus jitter


def _is_transient(exc: Exception) -> bool:
    """True for failures that a later identical request may survive."""
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError,
                        anthropic.RateLimitError,
                        anthropic.InternalServerError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        code = getattr(exc, "status_code", None)
        # 529 is Anthropic's overloaded status. A mid-stream overload can
        # surface without a usable status code, so match the payload too.
        if code is None or code >= 500:
            return True
        return "overloaded" in str(exc).lower()
    return False


def _stream_text(tag: str, pdf_path: str, client=None, **kwargs):
    """-> (streamed text, final message). Retries transient failures.

    Deliberately NOT catching TruncatedOutput or JSON problems: those are
    deterministic properties of the response and retrying them just buys the
    same answer twice. Only transport and capacity failures are retried.
    """
    name = Path(pdf_path).name
    for attempt in range(1, STREAM_RETRIES + 1):
        parts = []
        try:
            with (client or _client).messages.stream(**kwargs) as stream:
                for chunk in stream.text_stream:
                    parts.append(chunk)
                return "".join(parts), stream.get_final_message()
        except Exception as exc:
            if attempt == STREAM_RETRIES or not _is_transient(exc):
                raise
            delay = STREAM_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 3)
            print(f"[{tag}] {name}: {type(exc).__name__} "
                  f"({str(exc)[:70]}) after {len(''.join(parts))} chars - "
                  f"retry {attempt}/{STREAM_RETRIES - 1} in {delay:.0f}s",
                  file=sys.stderr)
            time.sleep(delay)


def _call(data: str, instructions: str, max_tokens: int, tag: str,
          pdf_path: str, effort: str = EFFORT) -> dict:
    """One API call returning parsed JSON, with truncation diagnostics.

    Text is accumulated off the stream as it arrives. Reading it back off
    stream.get_final_message() alone loses a text block that max_tokens cut
    mid-emission, which presents as an empty response with the full budget
    spent - indistinguishable from the model genuinely saying nothing.
    """
    streamed, resp = _stream_text(
        tag, pdf_path,
        model=MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        messages=[{
            "role": "user",
            # Document first and cache-marked, instructions second: the cached
            # prefix has to be identical across calls, and only the
            # instructions differ between A, B and the judge.
            "content": [pdf_block(data),
                        {"type": "text", "text": instructions}],
        }],
    )
    final = "".join(b.text for b in resp.content
                    if getattr(b, "type", None) == "text")
    raw = (final if len(final) >= len(streamed) else streamed).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    stop = getattr(resp, "stop_reason", None)

    dbg_dir = Path(__file__).parent / "data" / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    dbg = dbg_dir / f"{Path(pdf_path).stem}.{tag}.raw.txt"
    dbg.write_text(
        f"stop_reason={stop}\n"
        f"usage={getattr(resp, 'usage', None)}\n"
        f"blocks={[(getattr(b, 'type', '?'), len(getattr(b, 'text', '') or '')) for b in resp.content]}\n"
        f"streamed_chars={len(streamed)}\nfinal_chars={len(final)}\n"
        f"chars={len(raw)}\n\n{raw}"
    )

    if not raw:
        msg = (f"[{tag}] model returned no text (stop_reason={stop}). "
               f"See {dbg}")
        if stop == "max_tokens":
            raise TruncatedOutput(msg, tag, "", stop)
        raise ValueError(msg)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        s, en = raw.find("{"), raw.rfind("}")
        if s != -1 and en > s:
            try:
                return json.loads(raw[s:en + 1])
            except json.JSONDecodeError:
                pass
        if stop == "max_tokens":
            raise TruncatedOutput(
                f"[{tag}] output truncated at the token ceiling "
                f"({len(raw)} chars). Raw saved to {dbg}", tag, raw, stop
            ) from err
        raise ValueError(f"[{tag}] unparseable JSON (stop_reason={stop}, "
                         f"{len(raw)} chars). Raw saved to {dbg}") from err


def _components_split(data: str, pdf_path: str) -> dict:
    """Retry the component call as two smaller calls, one per table."""
    rows = []
    for scope, tag in ((_SCOPE_T1, "B_components_t1"),
                       (_SCOPE_T2, "B_components_t2")):
        try:
            out = _call(data, _INSTRUCTIONS_B + scope, MAX_OUT, tag, pdf_path)
            rows.extend(out.get("components") or [])
        except TruncatedOutput as exc:
            salvaged = _salvage_rows(exc.raw)
            if not salvaged:
                raise
            print(f"[{tag}] TRUNCATED - salvaged {len(salvaged)} complete "
                  f"row(s); the table will under-sum and reconciliation "
                  f"should flag this report", file=sys.stderr)
            rows.extend(salvaged)
    return {"components": rows}


def _normalise_systems(rows: list, pdf_path: str) -> list:
    """Model output -> exactly 12 rows, one per subcategory, in canonical order.

    The prompt asks for all twelve and the model usually complies, but a
    fixed-width feature matrix cannot depend on that: one report returning
    eleven rows turns systems.csv into a ragged table and every downstream
    join has to defend against it. Enforcing the shape here - where it is
    free and deterministic - is the difference between a schema and a hope.

    Three things get repaired:
      * a subcategory the model omitted becomes an unassessed row;
      * an unknown or misspelled `subcategory` is re-mapped through the
        taxonomy rules rather than discarded, so its costs survive;
      * duplicate rows for one subcategory are merged, summing the money and
        keeping the best condition, because dropping either copy would lose
        cost that reconciliation is about to check.
    """
    by_sub, unknown, orphans = {}, [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        flat = {f: row.get(f) for f in SYSTEM_FIELDS}
        sub = (flat.get("subcategory") or "").strip().lower().replace(" ", "_")
        if sub == SUBCATEGORY_OTHER:
            # Already known to belong to none of the twelve (the legacy
            # migration says so explicitly). Not an anomaly - don't log it.
            orphans.append(flat)
            continue
        if sub not in SUBCATEGORIES:
            # Don't throw the row away - it may be carrying money. Re-map it
            # from whatever the model did say, and record what happened.
            guess = classify_subcategory(
                " ".join(str(flat.get(k) or "") for k in
                         ("subcategory", "source_sections", "notes")))
            unknown.append((flat.get("subcategory"), guess))
            if guess == SUBCATEGORY_OTHER:
                # Belongs to none of the twelve. It is NOT dropped: it may be
                # carrying cost, and the systems block is summed against the
                # report's own stated totals. Dropping a costed row makes a
                # correct extraction fail reconciliation - measured, when this
                # function did drop them: 25 of 134 reports stopped tying out
                # on "reserves: property vs systems" purely because rows like
                # "Utilities" took their reserve dollars with them.
                flat["subcategory"] = SUBCATEGORY_OTHER
                orphans.append(flat)
                continue
            sub = guess
        flat["subcategory"] = sub
        if sub in by_sub:
            by_sub[sub] = _merge_system_rows(by_sub[sub], flat)
        else:
            by_sub[sub] = flat

    if unknown:
        print(f"[extract] {Path(pdf_path).name}: re-mapped {len(unknown)} "
              f"row(s) with an off-schema subcategory: {unknown[:5]}",
              file=sys.stderr)

    out = []
    for sub in SUBCATEGORIES:
        row = by_sub.get(sub)
        if row is None:
            row = {f: None for f in SYSTEM_FIELDS}
            row["subcategory"] = sub
            row["assessed"] = False
        elif row.get("assessed") is None:
            # The model filled the row but did not say. Content decides:
            # anything at all on the row means the report addressed it.
            row["assessed"] = any(
                row.get(f) is not None for f in
                ("condition", "condition_rating_numeric", "action_required",
                 "rul_years", "notes", "source_sections",
                 "immediate_repairs_usd", "short_term_repairs_usd",
                 "non_critical_repairs_usd", "replacement_reserves_usd"))
        out.append(row)

    # The 13th row, present only when it has to be. It holds whatever mapped
    # to none of the twelve AND carries money, so cost integrity survives the
    # regrouping. The feature matrix is the twelve: filter with
    # `subcategory != "other"` for modelling, keep it for any sum that has to
    # tie out against the report.
    if orphans:
        merged = orphans[0]
        for extra in orphans[1:]:
            merged = _merge_system_rows(merged, extra)
        if any(isinstance(merged.get(f), (int, float)) and merged[f]
               for f in _MONEY_FIELDS):
            merged["subcategory"] = SUBCATEGORY_OTHER
            merged["assessed"] = True
            out.append(merged)
    return out


_MONEY_FIELDS = ("immediate_repairs_usd", "short_term_repairs_usd",
                 "non_critical_repairs_usd", "replacement_reserves_usd")


def _merge_system_rows(a: dict, b: dict) -> dict:
    """Two rows for one subcategory -> one. Money adds, text joins.

    Summing rather than picking a winner: both rows are real costs the report
    stated, and reconciliation compares the total against the report's own
    stated total. Keeping one and dropping the other would under-sum and flag
    a correct extraction.
    """
    out = dict(a)
    for f in _MONEY_FIELDS:
        va, vb = a.get(f), b.get(f)
        vals = [v for v in (va, vb) if isinstance(v, (int, float))]
        out[f] = sum(vals) if vals else (va if va is not None else vb)
    for f in ("source_sections", "action_required", "notes"):
        parts = [str(v).strip() for v in (a.get(f), b.get(f))
                 if v not in (None, "")]
        # dict.fromkeys dedupes while keeping order
        out[f] = "; ".join(dict.fromkeys(parts)) or None
    for f in ("condition", "condition_secondary", "condition_rating_numeric",
              "rul_years"):
        out[f] = a.get(f) if a.get(f) is not None else b.get(f)
    # None, not False, when neither row said - otherwise the merged row
    # skips the content-based inference in _normalise_systems and a row
    # carrying real money comes out marked "not assessed".
    flags = [r.get("assessed") for r in (a, b)]
    out["assessed"] = True if any(f is True for f in flags) else (
        None if all(f is None for f in flags) else False)
    return out


@traceable(run_type="llm", name="extract_pca")
def extract(pdf_path: str) -> dict:
    """PDF path -> {"property": {...}, "systems": [...], "components": [...]}

    Two calls, not one. A single call has to emit ~50 property fields with
    grounding snippets AND every cost-table row, which overruns the output
    ceiling on large reports (Wickliffe: 177 pages, 27+ line items). Splitting
    keeps each response well clear of the limit and lets the component call -
    by far the biggest - use the whole budget on rows. If that still truncates,
    the component call is retried once as two per-table calls.
    """
    data, page_map = _prepare_pdf(pdf_path)

    try:
        a = _call(data, _INSTRUCTIONS_A, MAX_OUT, "A_property_systems", pdf_path)
    except TruncatedOutput:
        # Call A had no retry at all, so one over-long deliberation killed the
        # whole report after the PDF had already been paid for. The component
        # call has had a split retry since v3; this is the same idea for the
        # property call, and it costs nothing on reports that do not need it.
        print(f"[A_property_systems] truncated - retrying at effort="
              f"{RETRY_EFFORT!r}", file=sys.stderr)
        a = _call(data, _INSTRUCTIONS_A, MAX_OUT, "A_property_systems_retry",
                  pdf_path, effort=RETRY_EFFORT)
    try:
        b = _call(data, _INSTRUCTIONS_B, MAX_OUT, "B_components", pdf_path)
    except TruncatedOutput as exc:
        print(f"[B_components] truncated - retrying split by table",
              file=sys.stderr)
        try:
            b = _components_split(data, pdf_path)
        except TruncatedOutput:
            salvaged = _salvage_rows(exc.raw)
            if not salvaged:
                raise
            print(f"[B_components] split retry also truncated - salvaged "
                  f"{len(salvaged)} row(s) from the original response",
                  file=sys.stderr)
            b = {"components": salvaged}

    prop = {}
    for f in PROPERTY_FIELDS:
        cell = (a.get("property") or {}).get(f) or {}
        if not isinstance(cell, dict):
            cell = {"value": cell}
        page = cell.get("page")
        # The model numbered pages within the slice; report the original.
        if isinstance(page, int) and 1 <= page <= len(page_map):
            page = page_map[page - 1] + 1
        prop[f] = {"value": cell.get("value"), "page": page,
                   "snippet": cell.get("snippet"),
                   # No default. A missing confidence key means the model did
                   # not report one, which is unknown - not low. Defaulting to
                   # 0.5 put it under CONFIDENCE_FLOOR and nominated the field
                   # for the judge, so a report could arrive at judge_fields
                   # with 20 suspects purely from absent metadata. The judge
                   # re-reads the whole PDF to verify each one, which costs
                   # about what an extraction costs.
                   # collect_suspect_property_fields ignores non-numeric
                   # confidence, so None simply lets the other layers decide.
                   "confidence": cell.get("confidence")}

    systems = _normalise_systems(a.get("systems") or [], pdf_path)
    # Expand the sparse wire format back into the flat schema: the model
    # omits nulls and collapses the 12 year columns into a "years" dict,
    # which cuts its output several-fold and keeps long tables under the
    # output ceiling. Widening happens here, where it is free.
    components = []
    dropped_years = set()
    for row in (b.get("components") or []):
        flat = {f: row.get(f) for f in COMPONENT_FIELDS}
        years = row.get("years") or {}
        if isinstance(years, dict):
            for k, v in years.items():
                key = str(k).strip().lstrip("yYearr_ ")
                try:
                    idx = int(key)
                except ValueError:
                    continue
                # A calendar year that slipped through the prompt's
                # instruction to convert. Recording it as year_2024 would
                # invent a column; silently dropping it would make the row
                # under-sum. Neither - collect them and say so.
                if idx > RESERVE_YEARS:
                    dropped_years.add(idx)
                    continue
                if 1 <= idx <= RESERVE_YEARS:
                    flat[f"year_{idx}"] = v
        # Derived here, not asked for: deterministic, free, and traceable
        # to the rule that assigned it (taxonomy.explain). Recomputed by
        # validate.coerce_types after description coercion in case the
        # description changed.
        flat["subcategory"] = subcategory_for_component(flat.get("description"))
        if flat.get("rul_varies") is None:
            flat["rul_varies"] = False
        if flat.get("years_inflated") is None:
            flat["years_inflated"] = False
        components.append(flat)

    if dropped_years:
        print(f"[extract] WARNING {Path(pdf_path).name}: {len(dropped_years)} "
              f"year key(s) outside 1-{RESERVE_YEARS} were dropped "
              f"({sorted(dropped_years)[:6]}). Calendar years were not "
              f"converted to term years; those schedules will under-sum.",
              file=sys.stderr)

    return {"property": prop, "systems": systems, "components": components,
            "_prompt_version": PROMPT_VERSION}

# ── judge slice ────────────────────────────────────────────────────────────
# The judge's page budget. Distinct from the extraction slice on purpose.
#
# WHY: the judge re-sent the FULL extraction slice - a 90-page, ~121K-token
# document - to verify two to five scalar fields. Measured over 127 judged
# reports that was 15.3M input tokens, and because the extraction cache is
# written with a 5-minute TTL while a report takes longer than that to work
# through, roughly 40% of those sends missed the cache and paid 1.25x to
# rewrite the whole document. The judge was ~27% of the run for verdicts on a
# handful of cover-page facts.
#
# The fields the judge is actually asked about - report_firm, assessment_date,
# inflation_rate_pct, the reserve and repair totals, overall_condition, the
# subcategory conditions - live in the cover, the executive summary and the
# condition-summary table. So the judge gets the front block plus the pages
# extraction cited for the fields under review, and nothing else.
#
# The tradeoff is deliberate and worth stating: this slice cannot hit the
# extraction cache, because it is a different document. It is cheaper anyway -
# ~20 uncached pages beats ~90 pages at a 60% cache-hit rate by a wide margin -
# and it stops the judge's cost depending on how long the queue happened to be.
JUDGE_FRONT_PAGES = 14
JUDGE_MAX_PAGES = 26

# Raw-bytes ceiling for the judge slice, well under the API's request cap.
# Not a safety margin - a cost control. A cited page can land in a photo
# appendix: on the 239-page Canal Square report (46.9MB), the front 14 pages
# weigh 0.37MB while two arbitrary interior pages plus neighbours pushed the
# slice to 7.9MB. One mis-cited page would cost more than the whole-document
# send this slice exists to avoid, so heavy cited pages get dropped and the
# front block - which is where the judged facts live - is what survives.
JUDGE_MAX_BYTES = 4 * 1024 * 1024


@lru_cache(maxsize=8)
def _judge_pdf_b64(pdf_path: str, cited: tuple = ()) -> str:
    """-> base64 of a small slice: the front block plus the cited pages.

    `cited` is 1-based ORIGINAL page numbers, as stored on the extracted
    cells. Out-of-range values are ignored rather than raising: a page number
    is exactly the thing a bad extraction gets wrong, and the judge exists to
    catch that, so it must still run when the citation is nonsense.

    Each cited page brings its neighbours. A value read from the top of a
    table is often stated in the line above it, and a one-page window makes
    the judge answer "not stated" for something the report states plainly.
    """
    reader = PdfReader(pdf_path)
    total = len(reader.pages)

    keep = set(range(min(JUDGE_FRONT_PAGES, total)))
    for pg in cited:
        if not isinstance(pg, int):
            continue
        i = pg - 1
        if 0 <= i < total:
            keep.update(j for j in (i - 1, i, i + 1) if 0 <= j < total)

    front = set(range(min(JUDGE_FRONT_PAGES, total)))
    idxs = sorted(keep)[:JUDGE_MAX_PAGES]
    data, size = _encode(reader, idxs, pdf_path)
    if size <= JUDGE_MAX_BYTES:
        return data

    # Over budget. Weigh each cited page on its own and drop the heaviest
    # until it fits, never touching the front block. Photo pages weigh
    # megabytes and text pages weigh kilobytes, so this converges in a step
    # or two and removes exactly the pages that were not worth sending.
    def _page_bytes(i):
        w = PdfWriter()
        w.add_page(reader.pages[i])
        buf = io.BytesIO()
        w.write(buf)
        return len(buf.getvalue())

    extra = sorted((i for i in idxs if i not in front),
                   key=_page_bytes, reverse=True)
    dropped = []
    while extra and size > JUDGE_MAX_BYTES:
        heaviest = extra.pop(0)
        dropped.append(heaviest)
        idxs = [i for i in idxs if i != heaviest]
        data, size = _encode(reader, idxs, pdf_path)

    print(f"[judge-pages] {Path(pdf_path).name}: slice over "
          f"{JUDGE_MAX_BYTES / 1e6:.0f}MB - dropped {len(dropped)} heavy cited "
          f"page(s) {[i + 1 for i in dropped][:8]}; kept {len(idxs)} page(s) "
          f"at {size / 1e6:.1f}MB", file=sys.stderr)
    return data
