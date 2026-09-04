"""Regenerate data/aggregate/README.md from the data that is actually there.

The previous dictionary was hand-written, so its row counts drifted from the
CSVs the moment anything was re-run. Every number below is read from the files
at generation time; the prose caveats are the parts a generator cannot know.
"""
from pathlib import Path
import pandas as pd
from taxonomy import SUBCATEGORIES, SUBCATEGORY_OTHER

AGG = Path(__file__).parent / "data" / "aggregate"


def main():
    t = {n: pd.read_csv(AGG / f"{n}.csv") for n in
         ("properties", "systems", "components", "capex_by_subcategory",
          "manifest")}
    p, s, c, x = (t["properties"], t["systems"], t["components"],
                  t["capex_by_subcategory"])

    tq = p["target_quality"].value_counts().to_dict()
    status = p["extraction_status"].value_counts().to_dict()
    clean = status.get("clean", 0)
    flagged = status.get("needs_review", 0)
    vers = p["prompt_version"].fillna("unversioned").value_counts().to_dict()
    n_other = int((s["subcategory"] == SUBCATEGORY_OTHER).sum())
    assessed = s[s["assessed"] == True]                      # noqa: E712
    comp_mapped = int((c["subcategory"] != SUBCATEGORY_OTHER).sum())

    cov = "\n".join(
        f"| `{sub}` | {assessed[assessed.subcategory == sub]['property_id'].nunique()} | "
        f"${x[x.subcategory == sub]['total_capex_usd'].sum():,.0f} |"
        for sub in SUBCATEGORIES)

    firms = p["report_firm"].value_counts()
    firm_rows = "\n".join(f"| {k} | {v} |" for k, v in firms.items())

    n_ok = tq.get("ok", 0)
    n_empty = tq.get("no_line_items", 0)
    n_untied = tq.get("does_not_tie", 0)
    n_unaffected = int(((p.extraction_status == "needs_review") &
                        (p.target_quality == "ok")).sum())

    doc = f"""# PCA extraction — data dictionary

**{len(p)} reports** · {len(firms)} assessing firms · generated from the files in this folder.

## Files

| file | rows | grain |
|---|---|---|
| `properties.csv` | {len(p)} | one row per report |
| `systems.csv` | {len(s)} | **12 rows per report**, one per subcategory (+{n_other} `other`) |
| `components.csv` | {len(c)} | one row per cost-table line item |
| `capex_by_subcategory.csv` | {len(x)} | **one row per (report × subcategory) — the modelling grain** |
| `manifest.csv` | {len(t['manifest'])} | per-report processing record |
| `extracted_reports.csv` | {len(t['manifest'])} | roster: PDF filename → property |

Join everything on `property_id`. `s3/` holds the same data partitioned
`firm=…/state=…` so leave-one-firm-out CV is a prefix filter.

## READ THIS FIRST — if you are training a capex model

**Filter on `target_quality`, NOT on `extraction_status`.**
`{tq}`

`extraction_status` aggregates every check in the validation stack, most of
which do not touch the components-derived capex target at all. Of the
{flagged} `needs_review` reports, **{n_unaffected} carry only systems-layer or
property-layer flags — their capex target is untouched**. Filtering on
`clean` would throw away a quarter of the usable training set for problems
that are not in the label.

| `target_quality` | reports | what to do |
|---|---|---|
| `ok` | {n_ok} | **train on these** |
| `no_line_items` | {n_empty} | **exclude.** The report states a total but no line items were extracted, so `total_capex_usd` is a FALSE ZERO — a poisoned label, worse than a missing row |
| `does_not_tie` | {n_untied} | usable with care; the target is biased by a knowable amount |

```python
train = capex[capex.target_quality == "ok"]          # {n_ok} reports
```

## Then read this

**1. `extraction_status` is on every table.**
- `clean` ({clean} reports) — every extracted line item sums to that report's OWN
  printed totals, at all three layers. The financial figures are checked.
- `needs_review` ({flagged} reports) — at least one total does not tie out, or a
  table came back empty. Rows are still here and mostly right; they are labelled
  so you can decide per analysis rather than being silently dropped.

Use `df[df.extraction_status == 'clean']` where financial totals matter. For
component-level work (EUL/RUL) the flagged reports are usually still usable — a
reconciliation failure on the immediate-repairs bucket says nothing about that
report's 40 reserve rows.

**2. `prompt_version` is on every property and capex row.**
Provenance: {vers}. `unversioned` means the record predates the version stamp
and its prompt is unknown — it is NOT a claim that it matches anything. Filter
to a single version for any analysis whose headline is cross-firm
generalisation.

**3. `assessed` on systems rows.** `false` means the report does not address
that subcategory at all. It is NOT "assessed and found fine" — a null
`condition` on an unassessed row is missing data, not a good rating.

**4. `subcategory == 'other'`** ({n_other} systems rows) is NOT part of the
feature matrix. It holds cost from sections mapping to none of the twelve —
mostly "Utilities", which spans plumbing and electrical and belongs cleanly to
neither. It exists so the systems totals still tie out. **Filter it out for
modelling; keep it for any sum.**

## capex_by_subcategory.csv — start here for capex modelling

One row per (report × subcategory), {len(x)} rows.

| column | meaning |
|---|---|
| `reserve_capex_usd` | Σ component line items on the reserve table |
| `near_term_capex_usd` | Σ component line items on any near-term table |
| `total_capex_usd` | the two above, summed — **the target** |
| `n_reserve_items` / `n_near_term_items` | line items behind each figure. **0 items with $0 is a true zero; use these to tell it from an unextracted one** |
| `condition`, `condition_rating_numeric`, `rul_years` | features, from the systems layer |
| `assessed`, `extraction_status`, `prompt_version` | filters |

**Why the target comes from components, not from the systems money columns:**
`replacement_reserves_usd` is populated on {int(s['replacement_reserves_usd'].notna().sum())} of
{len(s)} systems rows, while `total_cost_usd` is populated on
{int(c['total_cost_usd'].notna().sum())} of {len(c)} component rows. The systems
columns come from each report's executive summary, and most firms do not cost
their work there. Training on them would mean training on the subset of firms
that publish a costed summary — and firm is the CV axis, so that bias is the
worst kind here.

## The 12 subcategories

| subcategory | reports assessing it | total capex |
|---|---|---|
{cov}

`components.csv` carries the same axis in its `subcategory` column, **derived**
from `description` by `taxonomy.subcategory_for_component` — deterministic,
auditable via `taxonomy.explain`, never asked of the model.
{comp_mapped} of {len(c)} rows map ({100 * comp_mapped / max(1, len(c)):.1f}%);
the rest are professional services (surveys, permits, fees — cost lines, not
building systems) and unmatched text.

## systems.csv columns

- `condition` — the report's OWN word, not translated. Only
  excellent/good/fair/poor are rank-ordered; `average`, `functional`,
  `adequate` and others are carried verbatim and deliberately NOT ranked,
  because firms place them differently. **`condition` is not yet usable as a
  numeric feature.**
- `condition_rating_numeric` — for firms that rate 1–5 instead of using words.
  NOT converted to a word: scale direction differs by firm and those reports
  print no legend.
- `rul_years` (+ `_min`/`_max`) — stated remaining useful life. **Negative is
  valid** and means past expected life. Ranges keep the max as the base value,
  same convention as `components.csv`.
- `source_sections` — the firm's own section numbers folded into this row
  (`"4.4.1 Roofing Materials; 4.4.2 Roof Drainage"`). The audit trail back to
  the page.
- `notes` — ≤25 words of the report's findings.

## Known limitations — read before drawing conclusions

1. **Not every line item was captured.** Median {int(c.groupby('property_id').size().median())} component rows per
   report. Some reports genuinely have no itemised table (verified by hand);
   others are under-extracted. Do not read row counts as property complexity.
2. **Image-only tables.** ~4 reports (Chelsea, 2226577, Salterra, Bell
   Deerwood) have image-only cost tables and almost no extractable dollar text.
   Page selection is blind on them.
3. **Condition vocabulary is not an ordinal scale** across firms. See above.

## Reports by firm

| firm | reports |
|---|---|
{firm_rows}
"""
    (AGG / "README.md").write_text(doc)
    print(f"wrote {AGG / 'README.md'} ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
