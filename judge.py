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

from extract import _sliced_pdf_b64, pdf_block, EFFORT

MODEL = "claude-sonnet-5"  # can use a stronger model here than extraction if you like
_client = anthropic.Anthropic()


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


@traceable(run_type="llm", name="judge_fields")
def judge_fields(pdf_path: str, record: dict, fields: list[str]) -> dict:
    if not fields:
        return {}
    subset = {f: record.get(f, {}).get("value") for f in fields}
    # same slice as extraction: full reports are 200+ pages / >32MB request cap
    data = _sliced_pdf_b64(pdf_path)
    instr = (
        "Re-read the attached PCA report and verify these extracted fields. "
        "For each, decide if the extracted value is correct per the report.\n\n"
        f"Fields to verify (name: extracted_value):\n{json.dumps(subset, indent=2)}\n\n"
        "Before overturning a value, note what these reports actually do - "
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
        "Return ONLY a JSON object mapping each field name to "
        '{"ok": true|false, "corrected_value": <correct value or null>, '
        '"reason": <short reason citing the report>}. '
        "If the extracted value is right, ok=true and corrected_value=null."
    )

    parts = []
    with _client.messages.stream(
        model=MODEL,
        # Thinking is on and it dominates this budget. Measured on Maybelle
        # Carter: 7,796 of 8,000 output tokens went to thinking, leaving 540
        # characters of answer - the JSON was cut off inside the second field
        # and all five verdicts were thrown away, taking a clean report to
        # needs_review over a verdict the model had partly produced. The floor
        # scales with the number of fields because each one carries a reason.
        max_tokens=min(32000, 12000 + 1500 * len(fields)),
        # Same reason as extract.EFFORT: thinking and answer share one
        # ceiling, and here thinking took 7,796 of 8,000 tokens.
        output_config={"effort": EFFORT},
        messages=[{
            "role": "user",
            # Document first and cache-marked so this send hits the cache
            # written by the extraction calls; the field list varies and must
            # come after it.
            "content": [pdf_block(data), {"type": "text", "text": instr}],
        }],
    ) as stream:
        for chunk in stream.text_stream:
            parts.append(chunk)
        resp = stream.get_final_message()

    streamed = "".join(parts)
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
        f"chars={len(raw)}\nn_fields={len(fields)}\nfields={fields}\n\n{raw}")

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
    verdicts = _salvage_verdicts(raw, fields)
    if verdicts:
        print(f"[judge] output truncated (stop_reason={stop}) - recovered "
              f"{len(verdicts)} of {len(fields)} verdict(s)", file=sys.stderr)

    # Fail SOFT on the remainder. The judge is a second-opinion layer, not the
    # extraction. If it can't answer, the affected fields stay unverified and
    # the report routes to needs_review - far better than discarding a
    # successful 12-minute extraction over a malformed verdict.
    for f in fields:
        verdicts.setdefault(f, {
            "ok": False, "corrected_value": None,
            "reason": f"judge returned no usable verdict "
                      f"(stop_reason={stop}); needs human review"})
    return verdicts