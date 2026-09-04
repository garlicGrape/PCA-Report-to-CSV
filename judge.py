"""
Layer 3: the LLM judge. A second model re-reads the PDF and checks ONLY the
fields flagged by layers 1-2 (failed grounding, out of range, low confidence).
Scoping it to suspects is the point — you get the "a model verified it's
correct" guarantee without paying to re-judge all ~200 fields on every report.

Returns {field: {"ok": bool, "corrected_value": <val|null>, "reason": str}}.

v3 changes:
- The PDF block is shared with extract.py and marked as a prompt-cache
  breakpoint, so this third send of the same document is a cache read at
  roughly a tenth of input cost instead of a third full-price upload.
- Streams instead of blocking. Same generation time, but a response cut off by
  max_tokens can no longer come back as an empty string: a truncated text block
  may lack its terminating event and get dropped from the accumulated final
  message, which is exactly the failure that made extract look like it had
  "returned no text" while spending its whole budget.
- The debug dump records usage and block types, so cache_read_input_tokens is
  visible without a LangSmith round trip.
- _sliced_pdf_b64 is memoised in extract.py, so asking for the slice here no
  longer re-scans and re-encodes the whole PDF.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
from langsmith import traceable

from extract import _judge_pdf_b64, pdf_block, EFFORT, _stream_text

MODEL = "claude-sonnet-5"  # can use a stronger model here than extraction if you like
_client = anthropic.Anthropic(max_retries=6)   # see extract.py


def _salvage_verdicts(raw: str, fields: list) -> dict:
    """Complete {field: verdict} pairs recoverable from truncated JSON.

    Scans for each field name as a key and brace-matches its object. A verdict
    only counts if its braces close and it parses - a half-written reason
    string is discarded, never guessed at.
    """
    out = {}
    for f in fields:
        i = raw.find(f'"{f}"')
        if i == -1:
            continue
        j = raw.find("{", i)
        if j == -1:
            continue
        depth, in_str, esc = 0, False, False
        for k in range(j, len(raw)):
            ch = raw[k]
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
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(raw[j:k + 1])
                        if isinstance(v, dict) and "ok" in v:
                            out[f] = v
                    except json.JSONDecodeError:
                        pass
                    break
    return out


# Which of a system row's fields the judge is shown and asked about. Not the
# whole row: `assessed` and the money columns are what a reviewer checks, and
# every extra key costs tokens in both directions.
_JUDGED_SYSTEM_FIELDS = ("condition", "condition_secondary",
                         "condition_rating_numeric", "rul_years",
                         "action_required", "source_sections",
                         "immediate_repairs_usd", "short_term_repairs_usd",
                         "non_critical_repairs_usd", "replacement_reserves_usd")


def _cited_pages(record: dict, fields: list[str]) -> tuple:
    """The original PDF pages extraction cited for the fields under review.

    These drive the judge's page slice. Only pages for the fields actually
    being judged - citing every field's page would rebuild the whole document
    and give back the cost this slice exists to save.
    """
    pages = {record.get(f, {}).get("page") for f in fields}
    return tuple(sorted(p for p in pages if isinstance(p, int) and p > 0))


@traceable(run_type="llm", name="judge_fields")
def judge_fields(pdf_path: str, record: dict, fields: list[str],
                 systems: list | None = None,
                 subcategories: tuple = ()) -> dict:
    """Verify suspect property fields and suspect subcategory rows in ONE call.

    Returns {key: {"ok", "corrected_value", "reason"}} where a key is either a
    property field name or "sys:<subcategory>". Property keys are unchanged
    from the previous contract, so existing callers keep working; the systems
    verdicts are additive.

    One call, not two. The expensive part of judging is putting the document
    in front of the model, so a second call to check the subcategory rows
    would roughly double the layer's cost to verify a handful of extra cells.
    They go in the same request, after the same document block.
    """
    systems = systems or []
    sys_keys = [f"sys:{c}" for c in subcategories]
    keys = list(fields) + sys_keys
    if not keys:
        return {}

    subset = {f: record.get(f, {}).get("value") for f in fields}
    by_sub = {r.get("subcategory"): r for r in systems}
    sys_subset = {
        c: {k: by_sub.get(c, {}).get(k) for k in _JUDGED_SYSTEM_FIELDS}
        for c in subcategories
    }

    # A targeted slice - the front block plus the pages extraction cited for
    # these fields - not the 90-page extraction slice. See the note on
    # JUDGE_FRONT_PAGES in extract.py for the measurement behind this.
    data = _judge_pdf_b64(pdf_path, _cited_pages(record, fields))
    instr = (
        "Re-read the attached PCA report and verify the extracted values "
        "below. For each, decide if it is correct per the report.\n\n"
        + (f"PROPERTY fields to verify (name: extracted_value):\n"
           f"{json.dumps(subset, indent=2)}\n\n" if fields else "")
        + "Before overturning a value, note what these reports actually do - "
        "the judge is here to catch extraction errors, not to standardise "
        "wording:\n"
        "- A condition may be the firm's own word ('average', 'satisfactory', "
        "'adequate', 'functional', 'marginal'). That is correct as extracted. "
        "Do NOT rewrite it onto excellent/good/fair/poor.\n"
        "- Hotels are counted in rooms or keys (num_rooms), senior housing in "
        "beds (num_beds), apartments in units (num_units). A null num_units "
        "on a hotel or a care facility is correct, not a miss.\n"
        "- num_stories, eul_years and rul_years may legitimately be ranges "
        "('1-4', '10-15') when a property has buildings of differing heights "
        "or a firm quotes a life span.\n\n"
        + (
            "\n\nAlso verify these CONDITION SUBCATEGORY rows. Each is a "
            "roll-up of the report's own sections onto a fixed 12-category "
            "scheme, so check the roll-up as well as the values:\n"
            f"{json.dumps(sys_subset, indent=2)}\n"
            "- source_sections lists the report sections that were folded "
            "into the row. Check they genuinely belong to that subcategory.\n"
            "- Site/parking lighting belongs to site_improvements, not "
            "electrical. Roof drainage and gutters belong to roofing; storm "
            "and site drainage to site_improvements. Retaining walls, fencing "
            "and signage to site_improvements. The emergency generator to "
            "electrical.\n"
            "- Costs must not be double counted across subcategories, and "
            "replacement_reserves_usd is the WHOLE-TERM total for that "
            "subcategory - a recurring item counts once per occurrence.\n"
            "- Report each of these under the key \"sys:<subcategory>\". A "
            "correction is the corrected ROW OBJECT (only the keys that "
            "change), not a scalar.\n"
            if subcategories else ""
        )
        + "\n\nReturn ONLY a JSON object with exactly these keys:\n"
        + f"{json.dumps(keys)}\n"
        + "Each key maps to "
        '{"ok": true|false, "corrected_value": <correct value or null>, '
        '"reason": <short reason citing the report>}. '
        "If the extracted value is right, ok=true and corrected_value=null."
    )

    # Same mid-stream retry as extraction: an overloaded_error can arrive
    # after the stream opens, where the SDK's max_retries does not reach.
    streamed, resp = _stream_text(
        "judge", pdf_path, client=_client,
        model=MODEL,
        # Thinking is on and it dominates this budget. Measured on Maybelle
        # Carter: 7,796 of 8,000 output tokens went to thinking, leaving 540
        # characters of answer - the JSON was cut off inside the second field
        # and all five verdicts were thrown away, taking a clean report to
        # needs_review over a verdict the model had partly produced. The floor
        # scales with the number of fields because each one carries a reason.
        max_tokens=min(32000, 12000 + 1500 * len(keys)),
        # Same reason as extract.EFFORT: thinking and answer share one
        # ceiling, and here thinking took 7,796 of 8,000 tokens.
        output_config={"effort": EFFORT},
        messages=[{
            "role": "user",
            # Document first, UNCACHED. This is the judge's own narrow
            # slice, sent once per report and reused by nothing, so a cache
            # entry would cost 1.25x to write and never be read back.
            "content": [pdf_block(data, cache=False),
                        {"type": "text", "text": instr}],
        }],
    )
    final = "".join(b.text for b in resp.content
                    if getattr(b, "type", None) == "text")
    raw = (final if len(final) >= len(streamed) else streamed).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    stop = getattr(resp, "stop_reason", None)

    dbg_dir = Path(__file__).parent / "data" / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    (dbg_dir / f"{Path(pdf_path).stem}.judge.raw.txt").write_text(
        f"stop_reason={stop}\n"
        f"usage={getattr(resp, 'usage', None)}\n"
        f"blocks={[(getattr(b, 'type', '?'), len(getattr(b, 'text', '') or '')) for b in resp.content]}\n"
        f"streamed_chars={len(streamed)}\nfinal_chars={len(final)}\n"
        f"chars={len(raw)}\nn_keys={len(keys)}\nkeys={keys}\n\n{raw}")

    def _parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise

    try:
        return _parse(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Truncated, but not necessarily empty. A cut-off response has usually
    # answered the first several fields completely before running out; the
    # old code discarded all of them and marked every field unverified, which
    # is how a report with one genuinely suspect field ended up flagged for
    # five. Recover whatever verdicts are complete and leave only the rest
    # unresolved.
    verdicts = _salvage_verdicts(raw, keys)
    if verdicts:
        print(f"[judge] output truncated (stop_reason={stop}) - recovered "
              f"{len(verdicts)} of {len(keys)} verdict(s)", file=sys.stderr)

    # Fail SOFT on the remainder. The judge is a second-opinion layer, not the
    # extraction. If it can't answer, the affected fields stay unverified and
    # the report routes to needs_review - far better than discarding a
    # successful 12-minute extraction over a malformed verdict.
    for f in keys:
        verdicts.setdefault(f, {
            "ok": False, "corrected_value": None,
            "reason": f"judge returned no usable verdict "
                      f"(stop_reason={stop}); needs human review"})
    return verdicts