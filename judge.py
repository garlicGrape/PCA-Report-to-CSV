"""
Layer 3: the LLM judge. A second model re-reads the PDF and checks ONLY the
fields flagged by layers 1-2 (failed grounding, out of range, low confidence).
Scoping it to suspects is the point — you get the "a model verified it's
correct" guarantee without paying to re-judge all ~200 fields on every report.

Returns {field: {"ok": bool, "corrected_value": <val|null>, "reason": str}}.
"""
import base64, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
from langsmith import traceable

from extract import _sliced_pdf_b64

MODEL = "claude-sonnet-5"  # can use a stronger model here than extraction if you like
_client = anthropic.Anthropic()


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
        "Return ONLY a JSON object mapping each field name to "
        '{"ok": true|false, "corrected_value": <correct value or null>, '
        '"reason": <short reason citing the report>}. '
        "If the extracted value is right, ok=true and corrected_value=null."
    )
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                {"type": "text", "text": instr},
            ],
        }],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
