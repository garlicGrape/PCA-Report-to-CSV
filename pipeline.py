"""
One report through the pipeline; outputs THREE CSVs per report:

  <name>_property.csv     one row  — property-level features
  <name>_systems.csv      ~20 rows — per-system condition + costs
  <name>_components.csv   many rows — Table 1 + Table 2 line items (EUL/RUL!)

Point it anywhere:
  python pipeline.py                     # processes data/inbox/
  python pipeline.py ~/pca-real-reports  # processes any folder of PDFs

Clean reports -> data/output/. Any unresolved issue -> data/needs_review/
with a .flags.json explaining exactly what failed.
"""
import csv, json, sys
from pathlib import Path

from langsmith import traceable

from schema import PROPERTY_FIELDS, SYSTEM_FIELDS, COMPONENT_FIELDS
from extract import extract
from validate import (deterministic_checks, reconciliation_checks,
                      grounding_checks, collect_suspect_property_fields,
                      completeness_checks)
from judge import judge_fields

HERE = Path(__file__).parent
DEFAULT_INBOX = HERE / "data" / "inbox"
OUTPUT = HERE / "data" / "output"
REVIEW = HERE / "data" / "needs_review"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _stamp_keys(record: dict, stem: str, pdf_path: str):
    """Put property_id + provenance on all three tables so they join."""
    pid = _property_id(stem)
    record["property"]["property_id"] = {"value": pid, "page": None,
                                         "snippet": None, "confidence": 1.0}
    record["property"]["source_file"] = {"value": Path(pdf_path).name, "page": None,
                                         "snippet": None, "confidence": 1.0}
    firm = record["property"].get("report_firm", {}).get("value")
    for row in record["systems"]:
        row["property_id"], row["report_firm"] = pid, firm
    for row in record["components"]:
        row["property_id"], row["report_firm"] = pid, firm
    return record


def _property_id(stem: str) -> str:
    """Stable, filesystem-safe id derived from the filename."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:60]


def _write_all(record: dict, stem: str, dest: Path):
    prop_row = {f: record["property"][f].get("value") for f in PROPERTY_FIELDS}
    _write_csv(dest / f"{stem}_property.csv", PROPERTY_FIELDS, [prop_row])
    _write_csv(dest / f"{stem}_systems.csv", SYSTEM_FIELDS, record["systems"])
    _write_csv(dest / f"{stem}_components.csv", COMPONENT_FIELDS,
               record["components"])


@traceable(name="pca_pipeline")
def process(pdf_path: str) -> dict:
    pdf_path = str(pdf_path)
    stem = Path(pdf_path).stem
    record = extract(pdf_path)
    record = _stamp_keys(record, stem, pdf_path)

    det = deterministic_checks(record)
    recon = reconciliation_checks(record) + completeness_checks(record)
    ground = grounding_checks(record, pdf_path)
    suspects = collect_suspect_property_fields(record, det, recon, ground)

    verdicts = judge_fields(pdf_path, record["property"], suspects)

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
                               "reason": v.get("reason", "flagged, no correction")})

    # table-level problems (reconciliation, table meta issues) always require
    # human eyes — a judge can't reliably repair a 60-row table.
    table_issues = recon + [i for i in det if i["where"] != "property"]
    status = "clean" if not unresolved and not table_issues else "needs_review"

    if status == "clean":
        _write_all(record, stem, OUTPUT)
    else:
        _write_all(record, stem, REVIEW)
        (REVIEW / f"{stem}.flags.json").write_text(json.dumps(
            {"deterministic": det, "reconciliation": recon, "grounding": ground,
             "property_suspects": suspects, "unresolved": unresolved}, indent=2))

    return {"id": stem, "status": status,
            "systems_rows": len(record["systems"]),
            "component_rows": len(record["components"]),
            "det_issues": len(det), "recon_issues": len(recon),
            "grounding_fails": len(ground), "unresolved": len(unresolved)}


if __name__ == "__main__":
    folder = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_INBOX
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        print("Usage: python pipeline.py [folder-of-pdfs]")
    for p in pdfs:
        print(process(str(p)))
