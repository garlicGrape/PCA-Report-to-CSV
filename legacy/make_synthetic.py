"""
Generate synthetic PCA reports (PDF) + matching gold labels (JSON).

Each report uses a different firm's *wording* for the same fields, so the
extractor has to do real label-mapping — the exact problem Jimmy flagged
about reports differing between firms. Gold values are canonical.

Run:  python make_synthetic.py --n 6
Output:
  data/inbox/<property_id>.pdf     the fake report
  data/gold/<property_id>.json     canonical values + the verbatim snippet
                                   and page each value should ground to
"""
import argparse, json, random
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

HERE = Path(__file__).parent
INBOX = HERE / "data" / "inbox"
GOLD = HERE / "data" / "gold"

# Three fictional firms, each with its own phrasing for the same fields.
FIRMS = ["Atlas Building Consultants", "Meridian PCA Group", "Southeast Facility Advisors"]

STYLES = getSampleStyleSheet()


def money(x):
    return f"${x:,.0f}"


def make_one(seed: int):
    rng = random.Random(seed)
    firm = FIRMS[seed % len(FIRMS)]
    pid = f"SYN-{1000 + seed}"

    roof_c = rng.choice(["fair", "good", "poor"])
    hvac_c = rng.choice(["fair", "good", "excellent"])
    vals = {
        "property_id": pid,
        "property_name": rng.choice(["Magnolia Court", "Cypress Landing", "Live Oak Senior Residences"]),
        "property_address": f"{rng.randint(100, 9999)} Peachtree St, Atlanta, GA",
        "report_firm": firm,
        "report_date": f"2026-0{rng.randint(1,9)}-1{rng.randint(0,9)}",
        "year_built": rng.randint(1968, 2015),
        "num_units": rng.randint(80, 420),
        "roof_condition": roof_c,
        "roof_rul_years": rng.randint(2, 25),
        "hvac_condition": hvac_c,
        "hvac_rul_years": rng.randint(3, 20),
        "repair_roof_usd": rng.randint(0, 400_000),
        "repair_hvac_usd": rng.randint(0, 300_000),
        "repair_other_usd": rng.randint(0, 200_000),
        "reserve_year_1_usd": rng.randint(50_000, 900_000),
    }
    vals["immediate_repairs_usd"] = (
        vals["repair_roof_usd"] + vals["repair_hvac_usd"] + vals["repair_other_usd"]
    )

    # Firm-specific phrasing -> the verbatim line that should appear in the PDF.
    # We record the snippet + page so grounding checks have a ground truth too.
    if seed % 3 == 0:
        lines = {
            "roof_condition":        f"Roof System — Overall Condition: {roof_c.title()}",
            "roof_rul_years":        f"Roof System — Remaining Useful Life: {vals['roof_rul_years']} years",
            "hvac_condition":        f"HVAC — Overall Condition: {hvac_c.title()}",
            "hvac_rul_years":        f"HVAC — Remaining Useful Life: {vals['hvac_rul_years']} years",
            "immediate_repairs_usd": f"Total Immediate Repairs: {money(vals['immediate_repairs_usd'])}",
            "reserve_year_1_usd":    f"Year 1 Reserve Recommendation: {money(vals['reserve_year_1_usd'])}",
        }
    elif seed % 3 == 1:
        lines = {
            "roof_condition":        f"Roof rating: {roof_c}",
            "roof_rul_years":        f"Roof RUL (yrs): {vals['roof_rul_years']}",
            "hvac_condition":        f"Mechanical/HVAC rating: {hvac_c}",
            "hvac_rul_years":        f"HVAC RUL (yrs): {vals['hvac_rul_years']}",
            "immediate_repairs_usd": f"Immediate Physical Needs total = {money(vals['immediate_repairs_usd'])}",
            "reserve_year_1_usd":    f"Reserve, Year 1: {money(vals['reserve_year_1_usd'])}",
        }
    else:
        lines = {
            "roof_condition":        f"The roof was assessed to be in {roof_c} condition.",
            "roof_rul_years":        f"Estimated remaining life of the roof is {vals['roof_rul_years']} years.",
            "hvac_condition":        f"HVAC equipment is in {hvac_c} condition overall.",
            "hvac_rul_years":        f"HVAC remaining life is approximately {vals['hvac_rul_years']} years.",
            "immediate_repairs_usd": f"Immediate repair costs total {money(vals['immediate_repairs_usd'])}.",
            "reserve_year_1_usd":    f"First-year reserve funding of {money(vals['reserve_year_1_usd'])} is advised.",
        }

    # Header lines that always appear (grounding for the identity fields).
    header = [
        f"PROPERTY CONDITION ASSESSMENT",
        f"Prepared by {firm}",
        f"Property: {vals['property_name']}  (ID {pid})",
        f"Address: {vals['property_address']}",
        f"Report Date: {vals['report_date']}",
        f"Year Built: {vals['year_built']}    Units: {vals['num_units']}",
    ]
    cost_lines = [
        f"Roof repairs: {money(vals['repair_roof_usd'])}",
        f"HVAC repairs: {money(vals['repair_hvac_usd'])}",
        f"Other immediate repairs: {money(vals['repair_other_usd'])}",
    ]

    # Build the PDF (single page keeps page numbers simple for grounding).
    INBOX.mkdir(parents=True, exist_ok=True)
    pdf_path = INBOX / f"{pid}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    for h in header:
        story.append(Paragraph(h, STYLES["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Systems Assessment", STYLES["Heading2"]))
    for k in ["roof_condition", "roof_rul_years", "hvac_condition", "hvac_rul_years"]:
        story.append(Paragraph(lines[k], STYLES["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Cost Summary", STYLES["Heading2"]))
    for cl in cost_lines:
        story.append(Paragraph(cl, STYLES["Normal"]))
    story.append(Paragraph(lines["immediate_repairs_usd"], STYLES["Normal"]))
    story.append(Paragraph(lines["reserve_year_1_usd"], STYLES["Normal"]))
    doc.build(story)

    # Gold: canonical value + the snippet/page a correct extraction should cite.
    snippets = {**{k: {"snippet": v, "page": 1} for k, v in lines.items()}}
    for k in ["property_id", "property_name", "property_address", "report_firm",
              "report_date", "year_built", "num_units",
              "repair_roof_usd", "repair_hvac_usd", "repair_other_usd"]:
        snippets[k] = {"snippet": None, "page": 1}  # present in header, snippet optional

    gold = {"values": vals, "grounding": snippets}
    GOLD.mkdir(parents=True, exist_ok=True)
    (GOLD / f"{pid}.json").write_text(json.dumps(gold, indent=2))
    return pid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    ids = [make_one(i) for i in range(args.n)]
    print(f"Wrote {len(ids)} synthetic reports -> {INBOX}")
    print(f"Wrote {len(ids)} gold files      -> {GOLD}")
    print("IDs:", ", ".join(ids))
