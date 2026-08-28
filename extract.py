"""
Layer 0: extraction, tuned to the real PCR structure.

Changes from v1:
- Slices the PDF to the first MAX_PDF_PAGES pages before sending (these
  reports run 200+ pages, but everything extractable — exec summary table,
  property description, Tables 1 & 2 — sits in the front; the rest is photo
  appendices. The API caps PDF requests at 100 pages / 32MB).
- Returns three layers: property (with per-field grounding), systems
  (exec summary table rows), components (Table 1 + Table 2 line items).
  Tables are validated by reconciliation instead of per-cell snippets.
"""
import base64, io, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
from langsmith import traceable
from pypdf import PdfReader, PdfWriter

from schema import (PROPERTY_FIELDS, SYSTEM_FIELDS, COMPONENT_FIELDS,
                    MAX_PDF_PAGES)

# Confirm the current model name in the Anthropic console; strings roll over.
MODEL = "claude-sonnet-5"

_client = anthropic.Anthropic()


def _sliced_pdf_b64(pdf_path: str, max_pages: int = MAX_PDF_PAGES) -> str:
    reader = PdfReader(pdf_path)
    if len(reader.pages) <= max_pages:
        return base64.standard_b64encode(Path(pdf_path).read_bytes()).decode()
    writer = PdfWriter()
    for page in reader.pages[:max_pages]:
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return base64.standard_b64encode(buf.getvalue()).decode()


_INSTRUCTIONS = f"""You are extracting structured data from a Property Condition
Assessment / Property Condition Report (ASTM E 2018 style). These reports have:
an EXECUTIVE SUMMARY TABLE (per-system condition ratings and costs), a property
description, narrative sections (2.1 through 5.2), and two cost tables:
TABLE 1 - IMMEDIATE REPAIRS and TABLE 2 - REPLACEMENT RESERVES.

Return ONLY a JSON object (no prose, no markdown fences) with exactly three keys:
"property", "systems", "components".

1. "property": an object with exactly these keys:
{json.dumps(PROPERTY_FIELDS)}
   Each key maps to: {{"value": <value or null>, "page": <1-based page or null>,
   "snippet": <verbatim supporting phrase or null>, "confidence": <0.0-1.0>}}
   - renovation_years / facade_materials / roof_types: join multiples with "; ".
   - fire_sprinklers / emergency_generator / basement: describe briefly or "none".
   - overall_condition: one of excellent/good/fair/poor (lowercase).

2. "systems": an array, one object per numbered row of the EXECUTIVE SUMMARY
   TABLE (2.1 Topography through 5.2 Fire Department), each with exactly:
   {json.dumps(SYSTEM_FIELDS)}
   - Ratings marked with two X's (e.g. Good and Fair): condition = the better
     rating, condition_secondary = the worse. One X: condition_secondary = null.
   - Blank cost cells are null, not 0. action_required "None" stays "None".

3. "components": an array, one object per line item of TABLE 1 (table:
   "immediate") and TABLE 2 (table: "reserve"), each with exactly:
   {json.dumps(COMPONENT_FIELDS)}
   - EUL/EFF AGE/RUL of "var"/"Varies": set the numeric field null and
     rul_varies true. Otherwise rul_varies false.
   - Table 1 rows: eul/age/rul/year_1..year_12 are null.
   - Table 2 rows: year_1..year_12 from the year columns; empty cells null.
   - Do NOT include the "Totals" rows as line items.

Rules for everything:
- Never invent values. Numbers as numbers (no "$"/commas). Dates YYYY-MM-DD.
- Include EVERY line item — completeness matters, totals will be checked
  against the report's stated totals.
"""


@traceable(run_type="llm", name="extract_pca")
def extract(pdf_path: str) -> dict:
    """PDF path -> {"property": {...}, "systems": [...], "components": [...]}"""
    data = _sliced_pdf_b64(pdf_path)
    with _client.messages.stream(
        model=MODEL,
        max_tokens=32000,   # Table 2 alone can be 50+ rows x 25 fields
        messages=[{
            "role": "user",
            "content": [
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf",
                            "data": data}},
                {"type": "text", "text": _INSTRUCTIONS},
            ],
        }],
    ) as stream:
        resp = stream.get_final_message()
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)

    # normalize the property layer so every field/cell shape exists
    prop = {}
    for f in PROPERTY_FIELDS:
        cell = (parsed.get("property") or {}).get(f) or {}
        if not isinstance(cell, dict):
            cell = {"value": cell}
        prop[f] = {"value": cell.get("value"), "page": cell.get("page"),
                   "snippet": cell.get("snippet"),
                   "confidence": cell.get("confidence", 0.5)}

    # normalize rows to their field lists
    systems = [{f: row.get(f) for f in SYSTEM_FIELDS}
               for row in (parsed.get("systems") or [])]
    components = [{f: row.get(f) for f in COMPONENT_FIELDS}
                  for row in (parsed.get("components") or [])]

    return {"property": prop, "systems": systems, "components": components}