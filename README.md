# PCA-Plus Extraction Agent (observability-first)

Turns PCA report PDFs into three schema-normalized CSVs per report (property, systems, components), tuned to the EBI/ASTM E 2018 report structure.

`systems` is a **fixed 12-row block per report** — one row per ASTM E 2018 subcategory (`taxonomy.SUBCATEGORIES`), so it is a feature matrix rather than a ragged list. It replaced a layer keyed on each firm's own section headings, which produced 3,771 rows carrying 1,216 distinct names and could not be pooled across firms. `components` keeps every line item (the timing layer's EUL/age/RUL training data) and carries a derived `subcategory` column on the same 12-value axis, so a line item and a condition rating finally join. Handles 200+ page reports by slicing to the extractable front section. Cross-table reconciliation validates the tables against the report's own stated totals.
as first-class concerns. LangSmith logs every step; a four-layer validation
stack catches extractions that are *wrong*, not just *missing* — including
qualitative fields like condition ratings.

## Validation stack (cheap → expensive)

0. **Extract** (`extract.py`) — Claude reads the PDF and returns, per field,
   `value + page + snippet + confidence`. Showing its work is what makes the
   later checks possible. Traced to LangSmith.
1. **Deterministic** (`validate.py`) — schema, types, ranges, categories, and
   totals reconcile to line items. Free. Catches structural/numeric errors.
2. **Grounding** (`validate.py`) — the cited snippet must actually appear on
   the cited page. Free. Catches hallucinated / mis-mapped values.
3. **Targeted judge** (`judge.py`) — a second model re-reads the PDF, but only
   for property fields that failed 1–2 or came back low-confidence, plus up to
   4 subcategory rows ranked worst-first. Scoped twice over: it judges a
   handful of values, not ~200, and it is sent the **front block plus the pages
   extraction cited** rather than the full 90-page extraction slice. Traced.

Clean reports → `data/output/*.csv`. Anything with an unresolved field →
`data/needs_review/*.csv` + a `.flags.json` explaining why. Nothing wrong slips
through silently.

**Eval** (`eval_suite.py`) — per-field accuracy against a labeled gold set.
That measured number is your enterprise proof and your faculty headline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in ANTHROPIC_API_KEY + LANGSMITH_API_KEY
```

Get keys: Anthropic → console.anthropic.com · LangSmith → smith.langchain.com

## Run it, synthetic-first

```bash
# 1) make fake reports (3 firms, different wording) + gold labels
python make_synthetic.py --n 6

# 2) run the full pipeline over data/inbox -> CSVs + review queue
python pipeline.py

# 3) measure per-field extraction accuracy against gold
python eval_suite.py
```

Open the run in LangSmith: you'll see one `pca_pipeline` trace per report with
nested `extract_pca` / `judge_fields` spans, latency, and token cost.

## Before real reports

1. **Paste your real ~200 fields** into `CANONICAL_FIELDS` in `schema.py`, and
   add their `FIELD_META` (type/range/category) + any `RECONCILE` groups. That
   list is your standardization layer across firms.
2. **Confirm the model name** (`MODEL` in `extract.py` / `judge.py`) in the
   Anthropic console — strings roll over.
3. **Hand-label ~10–15 real reports** as gold to get a trustworthy accuracy
   number. Synthetic proves the plumbing; real reports prove the model.
4. **Tune `CONFIDENCE_FLOOR`** in `schema.py` to trade judge cost vs. recall.

## Where n8n / Drive fit

This service can watch a local folder (as-is) or you keep a thin n8n Drive
trigger that drops PDFs into `data/inbox` and picks CSVs out of `data/output`.
The intelligence and observability live here in Python, not in n8n.

## Note for the team

The 8/27 plan had the agent feeding DynamoDB → S3 → Textract → SageMaker.
Output here is CSV (to Drive or DynamoDB). If you keep the DynamoDB path,
add a writer in `pipeline.py` where the CSV is written; if you go Drive,
the teammate's flow job needs Drive as its new source.
