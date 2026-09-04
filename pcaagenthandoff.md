# PCA-Plus Extraction Agent — Handoff
**Owner:** Sanchit Kumar (NYU Stern MSBAi capstone, PCA-Plus)
**Repo:** `/Users/sanchitkumar/MSBAi/Capstone/PCA Reports/pca-agent` (on GitHub)
**Last session:** 28–31 Aug 2026

> **For Claude Code:** read this whole file before touching anything. It is the
> authoritative state of the project as of 31 Aug 2026 — status, the fixes
> landed in v3/v4 and the 31 Aug corpus run, architecture, cache semantics,
> and open items.
>
> **The corpus is extracted and the CSVs exist. The live question is whether
> they are good enough to train on.**
>
> **Before spending ANY API budget, read "New tooling".** Most verification is
> now free (`verify_offline.py`), and the previous session overspent by ~35%
> re-running work that did not need re-running. Pay special attention to
> **Gotchas** before running or deleting anything.

---
## What this is
A Python service that turns Property Condition Assessment PDFs into three
schema-normalized CSVs, with LangSmith tracing and a validation stack. It
replaced an earlier plan (n8n + Lovable, then DynamoDB → S3 → Textract →
SageMaker) after Textract was found redundant: the LLM produces structured,
validated output directly.

**Two-layer model this feeds:**
- *Timing layer* — trains on `components.csv` (EUL / effective age / RUL per component). **The ~40-rows-per-report projection did not survive contact with the corpus:** the real median is 18, giving 2,409 rows across 134 reports, of which **1,336 carry the full EUL+age+RUL triple**. That is the timing layer's actual training set size.
- *Cost layer* — trains on `properties.csv` (n = number of reports).

Component physics generalizes across regions; unit cost does not. That split is the scalability argument for faculty.

---
## Current state (31 Aug 2026) — THE WHOLE CORPUS IS EXTRACTED

**131 of the 132 corpus reports have been extracted, plus the 3 originals that
are not in the inbox = 134 cached extractions.** The CSVs are built. The open
question is no longer "can we extract" but **"are the CSVs good enough to
train on"** — which is what the owner is working through now.

| | Count |
|---|---|
| PDFs in `data/inbox/PCA Reports/` | 138 files |
| Unique reports there (6 duplicates removed) | 132 |
| Plus originals delivered separately (no PDF in repo) | 3 |
| **Distinct reports total** | **135** |
| Extractions cached in `data/cache/raw/` | 134 |
| Never extracted | 1 — `7800 Alpha Road` |
| **Clean (all reconciliation passes)** | **95** |
| needs_review | 39 |

Outputs in `data/aggregate/`: `properties.csv` (134 rows x 62 cols),
`components.csv` (2,409 x 38), `systems.csv` (3,712), plus parquet and the
`firm=`/`state=` S3 layout. Every row carries `extraction_status`, so the 39
flagged reports can be included or excluded per analysis rather than being a
binary in/out decision.

### Is it complete? No — and here is exactly how it falls short

Three distinct kinds of incompleteness, which should not be conflated:

1. **One report missing.** `7800 Alpha Road` (93pp, watermark-only text layer)
   never extracted. It hit a 413 that is now fixed (see the base64 note in
   Gotchas) but was never re-run.
2. **Rows missing inside reports.** Median 18 component rows per report
   against a projection of ~40. Two reports returned **zero** components;
   24 returned fewer than 10. Some of that is real - Villa Oaks genuinely has
   no itemised table, verified by hand - but not all of it. The 39
   needs_review reports are flagged precisely because their line items do not
   sum to the report's own stated totals.
3. **Fields missing inside rows.** 1,336 of 2,409 component rows carry the
   full EUL + effective age + RUL triple. Mostly structural: Gabion publishes
   no life columns at all, Terracon publishes EUL only. **This missingness is
   correlated with FIRM**, so dropping incomplete rows drops whole firms -
   which is the axis leave-one-firm-out CV runs along. Decide the policy
   explicitly; do not `dropna()`.

**~4 reports can never be fixed by better prompting** - Chelsea, 2226577,
Salterra, Bell Deerwood have image-only cost tables and almost no extractable
dollar text, so page selection is blind on them.

### The thing that most needs fixing before this is a training set

The 134 extractions were produced under roughly **six different prompt
versions** over one session (01:08 to 22:02), because the prompt was being
fixed as new firms revealed new layouts. The 23 reports extracted before 19:00
never saw the inflation-deduplication fix, the narrative sweep, the
`total_cost_usd` semantics, or the firm-attribution fix.

That is not just incompleteness, it is **systematic bias**: extraction quality
correlates with when a report happened to run, and because reports were run in
firm-grouped batches, it correlates with firm. A model trained on this learns
partly from the order in which the pipeline was debugged. For a project whose
headline claim is leave-one-firm-out generalisation, that is the worst
possible confound.

**Recommendation: one final full re-extraction under a single frozen prompt**,
rather than piecemeal patching of the 39 failures. Measured costs for 132
reports:

| Configuration | Cost |
|---|---|
| Current settings | $88 |
| `--no-judge` (implemented) | **$64** |
| `--no-judge` + Batch API | $42 |
| `--no-judge` + Claude Haiku 4.5 | $32 |
| all three | ~$21 |

## The 12-subcategory consolidation (4 Sep 2026)

**What changed and why.** `systems.csv` was one row per numbered section, named
by whatever the assessing firm called it: **3,771 rows carrying 1,216 distinct
`system_name` strings** across 134 reports. "Exterior Walls" appears 55 times,
"Building Frame" 42, and behind them sits a thousand-long tail of one-offs.
That is not a feature — it is free text with a cost column attached, and it
cannot be pooled across firms, which is the entire premise of
leave-one-firm-out CV.

It is now **exactly 12 rows per report**, one per ASTM E 2018 subcategory,
defined once in `taxonomy.SUBCATEGORIES` / `SUBCATEGORY_SCOPE`:

    site_improvements  structural_frame_foundation  building_envelope  roofing
    mechanical_hvac    plumbing   electrical   vertical_transportation
    fire_life_safety   interior_elements   accessibility
    additional_considerations

`systems.csv` is therefore a fixed-width **134 x 12 feature matrix**. A
subcategory a report is silent on is a row with `assessed=false`, not a missing
row — "not assessed" and "assessed and found fine" are different facts and only
the flag separates them.

**What did NOT change.** `components.csv` keeps every line item and every
EUL / effective age / RUL triple. The timing layer's training set is untouched
at 2,409 rows / 1,336 complete triples. Components gained a **derived**
`subcategory` column on the same 12-value axis (`taxonomy.subcategory_for_
component`, run in `validate.coerce_types`), so a line item and a condition
rating finally join. Derived, never extracted: it costs no output tokens, is
identical on every re-run, and every row's bucket traces to the rule that
assigned it via `taxonomy.explain`.

**Provenance survived the fold.** Each row's `source_sections` carries the
firm's own section numbers and headings, joined with "; " — e.g.
`"4.4.1 Roofing Materials (BUR); 4.4.2 Roof Drainage"`. Without it the
regrouping would be unauditable.

**The migration was free.** `batch.migrate_legacy_systems` folds a cached
pre-subcategory record onto the twelve on load, so all 134 cached extractions
adopted the new schema without a single API call. `python verify_offline.py`
re-validates everything and rebuilds `data/aggregate/` at $0. Only future runs
pay the cheaper extraction.

**Mapping coverage.** 93.3% of the old system names land on one of the twelve.
The residue is deliberate, not a gap: `Utilities` / `Utility Providers and
Special Systems` (57 rows) describes who supplies water and power to the site,
spans plumbing and electrical, and belongs cleanly to neither — forcing it into
`electrical` would corrupt that feature on 57 rows to avoid an honest gap.
`Regulatory Compliance`, `Unit Mix`, and `Property Configuration` are not
building systems at all. These are dropped from the systems block and counted
in the coverage report `verify_offline.py` prints every run.

### Cost

The consolidation cuts call A's output; the judge changes cut its input. The
bill is dominated by putting the PDF in front of the model, which no schema
change touches:

**Measured with `count_tokens` over 6 reports** (free endpoint, no inference):
the extraction slice averages **134K tokens**; the new judge slice averages
**33K** — **0.25x**. Applied to the 127 reports that were judged:

| | Input tokens | Rate | Cost |
|---|---|---|---|
| Judge, before | 8.9M cache-write + 6.4M cache-read | $2.50 / $0.20 per MTok | **~$23.6** |
| Judge, after | ~4.2M uncached | $2.00 per MTok | **~$8.4** |

Roughly **$15 a run**, and — more useful than the number — the judge's cost no
longer depends on how long a report sat in the queue before reaching it.

Three changes got there, and the third was a bug in the first two:

| Lever | Effect |
|---|---|
| Judge sends front block + cited pages, not the 90-page slice | 0.25x the input tokens, measured |
| Judge verifies subcategory rows in the SAME call | second opinion on conditions at no extra document send |
| Judge's document block is `pdf_block(data, cache=False)` | its slice is sent once and reused by nothing; marking it cached is a straight 1.25x write penalty that never pays back |

The systems consolidation itself is a **data-quality** change first. It cuts
call A's output — 28 ragged rows become 12 — but call A's output is ~$14 of an
~$88 run and the property fields dominate it, so do not expect the schema
change alone to move the bill much. What it buys is a feature matrix that can
be pooled across firms.

**The judge's cache-miss was the real leak.** It re-sent the full extraction
slice to verify two to five cover-page facts, and because the extraction cache
is written with a 5-minute TTL while a report takes longer than that to work
through, roughly 40% of those sends paid 1.25x to rewrite a ~121K-token
document. Raising the TTL to 1h does NOT pay — the write premium on call A
(2x vs 1.25x) costs more than it saves on the judge. Sending a smaller
document does pay, and it makes the judge's cost independent of how long the
queue happened to be. See `JUDGE_FRONT_PAGES` in `extract.py`.

**Still on the table, not done here:** the Batch API is a flat 50% on
everything and nothing in this pipeline is latency-sensitive. It needs the
per-PDF flow (extract A -> extract B -> validate -> judge) restructured into
three corpus-wide phases keyed by `custom_id`.

---

## Working on this project (read first, in any editor)

**Setup**

    uv sync                 # or: pip install -r requirements.txt
    cp .env.example .env    # then fill in ANTHROPIC_API_KEY + LANGSMITH_API_KEY

**The three commands that matter**

    uv run python verify_offline.py           # re-validate everything, $0, ~2 min
    uv run python batch.py --aggregate-only --include-flagged   # write the CSVs
    uv run python batch.py "data/inbox/PCA Reports" --workers 3 --no-judge  # COSTS MONEY

**Rules of the road**
- **`data/cache/raw/` is the expensive asset.** ~$120 of extraction lives
  there. Never delete it. Three of those reports have no PDF in the repo at
  all, so they are irreplaceable.
- **Validator changed? Run `verify_offline.py`, not `batch.py`.** Deleting
  `data/cache/*.json` and re-running re-fires the judge on all 132 reports and
  re-uploads every PDF. That is what exhausted the API credits.
- **Nothing in `data/` is committed** and it must stay that way - the PDFs and
  every table derived from them carry client addresses and financials. See
  `.gitignore`; a `git add -A` used to stage 1,097 paths of client data and
  now stages 12 paths of source.
- **Check the account's usage limit before a long run**, not just the dollar
  cost. Two runs died mid-flight on caps.

**Using Cursor on this repo:** everything above applies unchanged - the
pipeline is plain Python with no editor-specific tooling. Point Cursor at this
file first; it is the design record. `schema.py` is the place to start reading
(it carries the cross-firm reasoning in comments), then `extract.py` page
selection, then `validate.py`. The comments explain *why* a rule exists and
which report forced it - that context is the expensive part and is not
recoverable from the code alone.

## New tooling (31 Aug) — read this before spending anything

### `verify_offline.py` — re-validate everything for $0
The single most useful thing added this session. Re-runs the ENTIRE validation
stack against the cached extractions with **no API calls**, and rewrites the
outer cache so `batch.py --aggregate-only` produces correct CSVs.

    uv run python verify_offline.py          # rebuild outer cache, ~2 min
    uv run python batch.py --aggregate-only  # then write the CSVs

Why it exists: `batch.py`'s outer cache stores the status computed when the
report ran, so every validator change makes every stored verdict stale. The
documented refresh (delete `data/cache/*.json`, re-run) **re-fires the judge on
all 132 reports**, re-sending every PDF a third time. That is the single most
expensive thing this pipeline can do, and it verifies only descriptive fields.
Doing that once cost roughly $25 and is what exhausted the API credits.

It also runs two checks the judge cannot:
- **`arithmetic_checks`** — `quantity x unit_cost` vs the stated extended cost,
  and `EUL - age == RUL`. Currently **123 rows** where a row's own numbers
  contradict each other.
- **`number_presence_checks`** — every headline dollar figure searched for in
  the PDF text as a human would see it written. Currently **14 figures appear
  nowhere in the text layer**.

Both are **advisory, not blocking** - they name rows to inspect. Promoting them
to blockers would discard most of the corpus on checks that are themselves
noisy (grounding alone has 372 mismatches, mostly extractor differences).

### `taxonomy.py` — component descriptions -> 16 categories
`description` is free text: **251 distinct strings in the first 258 rows, only
7 appearing more than once.** Nothing pools across firms without this. Now
maps **92.2%** of the full 2,409 rows deterministically (regex, auditable,
reproducible - `explain()` returns the category AND the text that triggered
it). Fit on 141 rows, so **re-fit the tail after any corpus change**;
`coverage_report()` exists for exactly that.

Two bugs worth remembering, both mine, both invisible without measurement:
a trailing `\b` after a stem defeats the stem (`refrigerat\b` misses
"Refrigerator", `floor\b` misses "flooring") and cost 20 points of coverage;
and `roof` swallows "rooftop", putting HVAC units in the roofing bucket - a
plausible-looking category with systematically wrong contents.

### `batch.py --no-judge`
The judge is ~a third of spend, re-sends the whole PDF, and second-opinions
only DESCRIPTIVE fields. Reconciliation - free - is what protects the money.
Measured: **$88 -> $64** for a full corpus run. Use it, then `verify_offline.py`.

### CSV quality pass (31 Aug) — all free, no API

Done while preparing the tables for the team to clean:

- **`state` was splitting regions.** "Florida" 23 times and "FL" 17 times, 33
  distinct values for ~25 states. `state` is the S3 partition key for
  leave-one-REGION-out CV, so Florida was two regions and a held-out Florida
  fold still had Florida in training. Now folded to two-letter codes via
  `US_STATES`: **33 -> 23 distinct.**
- **One count written into two denominator columns.** Eight senior-housing
  reports had identical `num_units` and `num_rooms`. Nulls the duplicate,
  keeping whichever matches `unit_basis` - and deliberately leaves the two
  genuinely mixed-use properties alone (King Edward really is a 250-unit
  apartment building AND a 186-room hotel).
- **The arithmetic check was crying wolf.** It flagged 121 rows; every sample
  inspected was an exact 2x, i.e. a component that recurs twice in the term -
  correct data reported as a fault. Now ignores integer multiples: **121 -> 41**,
  and the survivors are worth looking at.
- **Three "wrong" firm names were correct.** `Partners` (Solon, Ohio),
  `Property Solutions Inc.` (Chicago) and `Consulting Solutions Inc` (the CSI
  in "CSI Project 18-4338") are all real assessors named under "Prepared by".
  I had put Property Solutions in the CLIENT list by mistake, which would have
  discarded a correct attribution. **Check the cover page before "fixing" a
  firm name.**
- **`data/aggregate/README.md`** — a data dictionary written for the team:
  what `extraction_status` means, why `table` buckets must never be summed
  together, why `rul_years` can be negative, and the five known limitations.
  Hand this over with the CSVs.
- **`component_category`** is now a column in `components.csv` (16-way
  taxonomy, 92.2% classified), so pooling across firms does not require
  re-running `taxonomy.py`.

---
## Run history (30-31 Aug) — what each pass cost and taught

| Pass | Reports | Outcome |
|---|---|---|
| 7-report pilot, one per new shape | 7 | 5 clean; found 6 real bugs for $4.94 |
| Batch 1, 10 never-run firms | 19 | 10 clean -> 23/28 after free vocab fixes |
| Batch 2 | 107 | **hit the account API usage limit at report 55** |
| Resume after limit reset | 132 | all extracted; crashed writing parquet |
| Consistency pass (re-judge all) | 132 | **exhausted credits**; 289 judge failures |
| Offline revalidation | 134 | 95 clean, $0 |

**The staged approach paid for itself twice** - the pilot found six bugs for
$5 that would have cost $88 to discover on a full run, and batch 2's limit
crash was caught 55 reports in rather than at the end.

**Total session spend ~$120 against an $88 estimate.** The overrun was NOT the
corpus - it was ~40-50 re-extractions while iterating on fixes (Baldwin Park
alone ran four times) plus the 132-report consistency pass that re-judged
everything. Estimate the iteration, not just the run.

## The 7-report pilot (30 Aug) — what a real run found

Seven reports, one per new table shape, run through the full pipeline.
**$4.94 and ~12 minutes** (2,203 report-seconds across 3 workers). The cost
model built from the first four reports predicted $4.90, so per-report cost is
now a known quantity: **~$0.70, range $0.53–$1.07.** Full corpus of 132 should
land near **$90–100**.

| Shape | Report | Result |
|---|---|---|
| CBC | Anatole Daytona | clean |
| EPIC | Aladdin Hotel | clean |
| AEI 5-yr | Homewood Suites HP | clean |
| LandScience | 21961077 | clean |
| Terracon | ARIUM Apartments | clean |
| Gabion | Baldwin Park | needs_review → fixed |
| Tetra Tech | Maybelle Carter | needs_review → fixed |

**What the offline work got right:** both hotels returned `num_rooms` (193 and
124 keys) with `unit_basis="rooms"`. Reserve terms came back 5, 5, 10, 10, 10 —
the 12-year assumption really was wrong. Terracon's range EULs coerced as
designed. Gabion produced 57 component rows from a table with no EUL, EFF AGE
or RUL column at all.

**What only a real run could find — six fixes:**

1. **`total_cost_usd` does not mean what the "Total Cost" column says.** Tetra
   Tech's asphalt row prints TOTAL COST $17,920 and then bills $17,920 in year
   2 *and* year 7 — the row costs $35,840 over the term and $17,920 is one
   occurrence. Reconciliation caught it exactly as designed. The prompt now
   defines total_cost_usd as the term total, makes the year cells
   authoritative when they disagree with the printed column, and sends the
   per-occurrence figure to `cycle_replace_cost_usd`.
2. **Gabion has no total column at all**, so 52 of its 57 rows came back with
   `total_cost_usd` null — a table that cannot be reconciled against anything.
   Now derived from the row's year cells, which is what Gabion's own "Total
   Expenditures" figure adds up to.
3. **Gabion's headline reserve figure is a present value.** "Total Present
   Value (With Contingency)" is discounted and can never equal a sum of line
   items, so it must not be `reserves_total_uninflated_usd`. It is a real
   number the report states, so it got its own column
   (`reserves_total_present_value_usd`) rather than being forced into one that
   lies. Same rule for combined "including immediate needs" totals: leave the
   reserve total null rather than use a number that includes another bucket.
4. **`non_critical` was a real category, not a hallucination.** Tetra Tech
   divides costs three ways — Immediate **Critical** (health and safety),
   Immediate **Non-critical**, and reserves — each its own numbered section
   with its own total. The schema allowed three table values and rejected the
   report. Added as a fourth, with a matching
   `non_critical_repairs_total_usd` and its own reconciliation pair. It is
   **not** the same axis as `short_term`: short_term is about WHEN money is
   spent, non_critical about WHY, and an underwriter treats them differently.
5. **The judge was truncating and discarding its own work.** On Maybelle
   Carter, 7,796 of its 8,000 output tokens went to thinking, leaving 540
   characters — the JSON was cut off inside the second field and all five
   verdicts were thrown away, flagging five fields the model had partly
   answered. `max_tokens` now scales with the field count, and
   `_salvage_verdicts` recovers every verdict whose braces close. This is
   firm-independent and would have hit any report with several suspects.
6. **Firm attribution was wrong on 2 of 7 — in the *clean* reports.** An AEI
   report came back as "Bridge House Advisors Corp." (the lender) and an EPIC
   report as "Partners": the model read "Prepared for" instead of "Prepared
   by". **Nothing flagged either**, because a wrong firm name is still a valid
   string — and report_firm is the S3 partition key and the unit of the
   leave-one-firm-out split the whole generalisation argument rests on. Two
   changes: the prompt now says the assessing firm is the one repeated in the
   running header/footer while the client appears once on the cover, and
   `validate._client_named_as_firm` flags a report_firm matching a known
   client/lender so the judge re-reads the cover instead of the value
   standing.

7. **Thinking was consuming the entire output budget and killing reports.**
   This is the biggest single finding of the pilot and it only appeared on the
   *second* pass, after the prompt grew. Baldwin Park's call A came back
   `stop_reason=max_tokens`, `thinking_tokens=32000` of 32000,
   `blocks=[('thinking', 0)]` — the model deliberated through the whole budget
   and emitted **no text at all**, which surfaces as "model returned no text"
   after the PDF has already been paid for. `max_tokens` is a ceiling on
   thinking AND answer together and nothing reserved room for the answer, so
   raising it would only have made the same failure more expensive.
   Extraction is transcription of already-structured tables, not open-ended
   reasoning, and the depth was never earning its cost: measured thinking
   share was 32,000/32,000 on that call, 23,313/31,215 on a call B, and
   7,796/8,000 in the judge. Setting `output_config={"effort": "medium"}`
   (`extract.EFFORT`, shared by `judge.py`) bounds it. Measured on the
   1 Canal Square Plaza canary: call A thinking fell 6,993 → 1,365, the report
   stayed **clean**, and it ran **faster — 107s vs 153s**. Call A also had no
   retry at all, so one over-long deliberation killed the whole report; it now
   retries once at `effort="low"`, the same way the component call has had a
   split retry since v3.
8. **The systems layer needed the same two fixes as the components layer.**
   Once Maybelle Carter's component tables tied out exactly, its remaining two
   failures were both one layer up: systems summed 36,175 against a stated
   immediate total of 3,750 — exactly 3,750 critical + 32,425 non-critical,
   because the systems layer had nowhere to put a non-critical cost and folded
   everything into `immediate_repairs_usd`. Its reserve sum was short by
   17,920, the same asphalt row counted once instead of twice. `SYSTEM_FIELDS`
   now carries `short_term_repairs_usd` and `non_critical_repairs_usd` with
   matching reconciliation pairs, and the systems prompt carries the
   term-total rule. **Lesson for the next firm: a bucket or a semantic added
   to the components layer has to be added to the systems layer too, or the
   cross-table check fails on the layer you didn't change.**
9. **Gabion's condition vocabulary is about compliance and age, not wear.**
   "non-compliant", "dated", "obsolete", plus "unknown" and "not performed".
   The first three are carried unranked for the same reason as "average": a
   dated kitchen can be in good condition, and "non-compliant" is a code
   finding, not a position on an excellent-to-poor axis. "unknown" and "not
   performed" fold to `na` — they are the existing "not assessed" statement in
   different words.

10. **Contingency markup — a stated total the line items can never reach.**
   Baldwin Park's "missing $4,300" was not a missing row. Gabion prints the
   arithmetic in its own table footer: `Subtotals $43,000` /
   `Contingency: 10.0% $4,300` / `Escalated Totals: $47,300`. The five
   immediate line items really do sum to $43,000 and the stated total really
   is $47,300. Chasing the gap as an extraction bug was chasing a number that
   was never in the table. Added `contingency_pct` / `contingency_usd`, and
   `reconciliation_checks` now accepts `stated == summed + contingency` — but
   ONLY when the report states a contingency, so a genuinely missing row still
   fails. **Expect this at other firms: check the table footer before treating
   a small fixed gap as a lost row.**
11. **Gabion has five row classes, not two.** I.N. (Immediate Needs),
   R.R. (Repairs and Replacements), **N.C. (No Cost Items** — real findings
   with no capital attached), **F.O. (Functional Obsolescence)** and
   **Q.D. (Questionable Durability)**. Their legend is on the page after the
   table. The immediate column is not just I.N. rows: an F.O. and an R.R. row
   with start-year 0 also land there, which is how five rows sum to $43,000
   when only three carry the I.N. prefix.

### Where the pilot ended: 9 clean, 2 needs_review

After the fixes, **every deterministic and category error across all 11
processed reports is gone.** What remains is two reports carrying small,
specific reconciliation gaps — which is reconciliation doing its job, not a
bug to paper over:

- **Baldwin Park (Gabion)** — stated immediate total 47,300 vs 43,000 summed.
  Both layers now agree with *each other* and disagree with the report, so it
  is one missing I.N.-class row, not a mapping error. All 57 component rows
  now carry a total (was 5 of 57), the present value is in its own field, and
  the reserve reconciliation is gone because the report states no undiscounted
  reserve total.
- **Maybelle Carter (Tetra Tech)** — component layer ties out **exactly** on
  all three buckets (3,750 / 32,425 / 496,486) and the systems immediate split
  is now correct. One gap left: systems reserves sum 478,566 vs 496,486, the
  same recurring asphalt row counted once instead of twice at the systems
  layer.

Both need a human eye on one row each. Neither should be "fixed" by loosening
a check.

**Verified NOT a problem:** CBC's Anatole Daytona returned 7 immediate-only
components with a null reserve total, which looks like under-extraction. That
report genuinely has one cost page and no reserve table. Checked before
changing anything.

---
## What was wrong, and what fixed it (30 Aug — the cross-firm pass)

Everything below was found by reading all 132 reports offline, without spending
an API call. None of it would have surfaced as an error at run time; most of it
would have surfaced as *missing rows*, which is the failure mode this pipeline
is least able to see.

**1. Page selection was a vocabulary list, and vocabulary is what a new firm
does not share.** `_TABLE_MARKERS` was written from four firms' column headers.
Four firms in the wider corpus carry perfectly ordinary cost tables that scored
**zero** on it: Gabion ("Present Worth / Start Year of Occurrence" — no EUL, EFF
AGE or RUL column anywhere), LandScience and CBC (costs itemised inside a
section table), Tetra Tech ("EXPECTED LIFE / REFLECTIVE AGE / REMAINING LIFE").
Their words are in the marker list now, which fixes those four and nothing else.
What generalises is `_structural_score`: **currency density** and **runs of
consecutive calendar years used as column headers**. A page dense in dollar
amounts is a cost table whatever it calls its columns. That is the part that
will catch firm 17.

**2. The blind first-N fallback was firing on the good case.** It triggered
whenever `_compose` returned no *tail* pages — but that also happens when every
table page sits inside the front block, which is exactly what you want. Ten
reports were paying for a 90-page send on every call to re-cover a front block
already contained in 30. The fallback now fires only when no page anywhere
scores, which in this corpus is one report.

**3. A slice over the size limit was sent anyway and rejected by the API.**
`_prepare_pdf` trimmed the front block to its floor, printed a warning, and
returned the oversized slice regardless. 7800 Alpha Road (93pp, 33.8MB) would
have failed on every attempt. Selection is now size-aware end to end: once the
front block is at its floor it drops its lowest-scoring pages until the request
fits, and says which cost-table pages it had to give up. All 138 files now
produce a valid request (largest slice: 31.4MB against a 32MB cap).

**4. Reserve terms are not 12 years.** They run 5 (AEI's Homewood Suites set),
10 (NV5, Terracon, Tetra Tech, Lender Consulting, Gabion), 12 (most), and 15
(Partner's German Church report). Emitting only `year_1..year_12` silently
dropped German Church's last three years — and because the schedule then no
longer summed to the row's own total, it would have looked like a *bad
extraction* rather than a short schema. `RESERVE_YEARS = 15`; the column names
are derived from it, and a year key outside the range is now reported rather
than dropped.

**5. Component and system numerics were never coerced — only flagged.**
`coerce_types` ran on the property layer alone. The two layers that carry the
training data got a deterministic type error and were then written to CSV as
strings anyway. This is routine, not exceptional: Terracon quotes EUL as "5-7"
and "20-25", AEI quotes expected life as "10-15" with remaining life "1-2".
Ranges now split into min/max the same way `num_stories` does, base field takes
the max, and the report is not sent to review over a value that was perfectly
readable.

**6. "Varies" was being counted as a coercion failure.** It is a *statement* —
the firm saying the component's remaining life differs across the property,
which the schema already carries as `rul_varies`. Tetra Tech writes it on a
quarter of its rows. Flagging it would have dropped those reports out of the
aggregate for saying something true. `_NON_VALUES` now separates "there is no
number here" from "this did not parse".

**7. The condition vocabulary was too narrow to survive the corpus, and the
obvious widening would have been wrong.** A rating outside `CONDITIONS` fails
the category check and takes the whole report to needs_review, so the list has
to be generous. But a rating inside `CONDITION_ORDINAL` becomes a *number* in
the timing model, so that one has to be strict. The corpus decides it:

  * **"New" folds to excellent.** Most legends in this corpus define it that
    way in their own words — "Excellent: New or like New" (CBC, Nova, Atwell,
    LandScience, Gabion), "1 - Excellent  New or like new condition" (The
    Carlisle Naples). That is reading the firm's definition, not a judgement.
  * **"Average" must NOT be ranked.** The Carlisle Naples prints a 1-5 legend
    where Average is its own level *between* Good and Fair. CBC, Nova and
    Atwell print a legend defining **Good** as "average to above-average
    condition". Same word, two positions, depending on who wrote the report.
  * **"Adequate", "acceptable", "serviceable", "satisfactory"** join
    "functional" as things that assert the system *works* without placing it
    on the wear axis. Carried verbatim, ranked by nobody. This is the v3
    "functional" decision applied consistently rather than one exception.
  * **"Good to Fair"** now splits into `condition` + `condition_secondary`,
    which is where the schema always intended it to go. It only splits when
    *both* halves are recognisable ratings — "roof to be replaced" stays one
    string and still fails loudly.

**8. Six of the 138 files are duplicates, and three of them do not share a
property name.** "23 - PCA Report 2021.pdf" is byte-identical to "River Bend
Assisted Living & Memory Care"; "Village Cove Assisted Living" is byte-identical
to "Brookdale Hilton Head Village"; "2226477" is "Lofts at the Highlands"
re-exported. Each one is a second extraction of a report already paid for and —
the part that does not show on the invoice — two identical properties in the
training tables, which inflates n, correlates rows, and **silently breaks
leave-one-property-out CV**: the held-out property is in the training fold under
another name. `batch.py` now dedupes on file hash then cover-page text, keeps
the more descriptive filename, and prints what it skipped. `--keep-duplicates`
overrides.

**9. Firm names were free text, and the S3 layout partitions on them.**
"EBI Consulting" / "EBI" / "EBI Consulting, Inc." are three partitions of one
firm, and leave-one-firm-out CV then leaks — the held-out firm is still in the
training fold under a different spelling. That is the project's headline
generalisation claim, so `report_firm` is now folded onto a canonical name from
`schema.FIRM_PATTERNS`, one registry shared by the pipeline and the profiler. An
unrecognised firm keeps its extracted name rather than becoming UNKNOWN —
naming it is what makes it countable next time.

**10. The asset class changed under the schema.** The corpus is heavily
hotel-weighted (Hampton Inn, Courtyard, Embassy Suites, DoubleTree, Homewood,
Crowne Plaza, Residence Inn, Staybridge...), and hotels are counted in **rooms
or keys**, not units. There was nowhere to put that, so it would have landed in
`num_units` and quietly corrupted every per-unit cost metric. Added `num_rooms`
and `unit_basis` (units/beds/rooms/sf), and the judge is told not to "correct" a
null `num_units` on a hotel.

**11. Gabion's schedule is inflated, so it cannot reconcile.** Their year
columns escalate year over year — a $51,500 item shows $65,239 later — so a
*correct* extraction of one of their reports would fail the Σyears ≈ total check
on every single row. Rows now carry `years_inflated`, and reconciliation skips
those rather than reading a correct extraction as a miscount.

**Also:** the profiler's shape classifier was matching squashed regexes, which
silently rewrote them (`r-total|cumulative r ?-` squashed to `rtotal|cumulativer`,
and `rtotal` is a substring of `yeartotal` — 21 reports landed in Terracon's
two-report shape). It matches literal phrases now, with a minimum length before
the squashed form is trusted. Unclassified reports: 68 → 23.

---
## What was wrong, and what fixed it (29 Aug)

**1. Wickliffe `stop_reason=max_tokens` with zero text.**
`_call` read output only from `stream.get_final_message()`. A text block cut
off by `max_tokens` can lack its terminating event and be dropped from the
accumulated message, so a truncated response presented as "model returned no
text". Fixed by accumulating from `text_stream` and keeping whichever copy is
longer. Ruled out first, in order: extended thinking (not enabled,
`thinking_tokens=0`), tool/structured-output blocks (none), model-string
rollover (`claude-sonnet-5` returns `['text']` normally).

**2. The first-50-page slice was dropping table content silently.**
Wickliffe is 177 pages; pages **169–172** carry cost-table markers and were
never sent. This produces missing rows, not an error, and no validation layer
can distinguish it from a report that genuinely had fewer line items. The
docstring assumption that everything extractable sits in the front does **not**
hold for Bureau Veritas. `extract.py` now scores every page for cost-table
markers, keeps a front block plus the pages that actually carry tables, and
warns on stderr when marker pages don't fit the budget. `MAX_PDF_PAGES` raised
50 → 90 (API hard cap is 100 pages / 32MB; both enforced, front block auto-trims
on size).

**3. `_num` concatenated every digit in a string.**
`"44,495 (per the rent roll dated August 02, 2023)"` parsed as
**44,495,022,023**. Caught only because the range check happened to reject it —
any garbage landing inside a plausible range would have entered the training
data. Now takes the first well-formed number.

**4. `coerce_types` returned successful coercions as issues.**
`"444 units"` → `444` is a correct recovery and it was flagging the report.
Recoveries now log to stderr; only genuine nulls come back as flags.

**5. `num_stories` ranges were nulled, excluding the report.**
`"1-4"` (BV) and `"One and two floors"` (EMG) are facts about properties with
buildings of differing heights. Now parsed into `num_stories_min` /
`num_stories_max`, base field gets the max. Range detection is narrow: explicit
digit ranges or word-spelled numbers only.

**6. Condition vocabulary across firms.**
Partner writes "not applicable" where EBI ticks NA. `CONDITION_SYNONYMS` folds
exact synonyms onto the canonical scale. **"functional" was deliberately NOT
mapped** — it says the thing works, not how worn it is, and mapping it to
good/fair would invent precision the report doesn't contain. It is carried as
its own value and excluded from the new `CONDITION_ORDINAL`, so it cannot
silently become a number in the timing model. Open decision.

**7. Grounding was measuring the wrong thing.**
The model and pdfplumber parse the PDF with different extractors, so exact
matching failed on text plainly present — proof: the model quoted
`"wasobserved"`, a text-layer artifact with the space already lost. Three
failure modes: model-inserted ellipses (can never match verbatim), table rows
flattened to single spaces, and encoding (ç, en-dashes, smart quotes). Now:
unicode folding, ellipsis-aware ordered fragment matching, and a
punctuation-stripped fallback. Also stopped re-normalizing the whole document
once per property field (~50 passes over a 177-page report per validation).

**8. `batch.py` dropped every `type` issue, not just property ones.**
`[i for i in det if i["kind"] != "type"]` discarded component and system type
failures that coercion never touches — a non-numeric `total_cost_usd` on a
Table 2 row was flagged and then silently thrown away. Now the deterministic
checks are simply re-run after judge + coercion, since values have changed
twice by that point.

**9. `_ensure_property_fields` (crash prevention).**
`data/cache/raw/` holds extractions made against the older `PROPERTY_FIELDS`.
Adding `num_stories_min` / `num_stories_max` meant every reused extraction was
missing them, and `aggregate()` indexed `record["property"][field]` directly —
`KeyError` on reports that worked yesterday. New firms will keep forcing new
fields, so this backfills rather than invalidating the expensive cache.

**10. Also:** stale `.flags.json` files are now deleted when a report goes
clean; `--aggregate-only` on an empty cache prints guidance instead of a pandas
`KeyError`; `sum(...) or None` no longer treats a real 0.0 total as
"nothing to compare"; completeness now catches rows that exist but carry no
`total_cost_usd`.

**Not a bug:** the doubled `[pages]` line is `judge_fields` sending the PDF for
suspect fields, which is the documented design.

---
## Architecture
```
PDF → select pages → 2 LLM calls → validate → judge → 3 CSVs
                            ↓
                     LangSmith traces
```

| File | Role |
|---|---|
| `schema.py` | Canonical field lists + type/range metadata. The standardization layer across firms. |
| `extract.py` | Page selection, two API calls (A: property + systems, B: component cost tables), truncation recovery. |
| `validate.py` | Deterministic checks, coercion + category normalization, completeness, reconciliation, grounding. |
| `judge.py` | Targeted LLM second opinion on suspect property fields only. |
| `batch.py` | Parallel folder processing, two-level cache, aggregation, S3 layout. |
| `pipeline.py` | Single-report path. |
| `verify_offline.py` | **Re-runs the whole validation stack with NO API calls** and rewrites the outer cache. Use after any validator change instead of deleting the cache and re-running. |
| `taxonomy.py` | Component `description` -> 16 canonical categories, deterministic and auditable. 92.2% coverage. |
| `profileCorpus.py` | Offline survey of a PDF folder — firms, table shapes, page budget, duplicates. Imports the pipeline's own markers and registry, so its answer predicts the real run. No API calls. |
| `smoke.py` | LangSmith connectivity check. |

**Run commands:**
```bash
uv run python batch.py ~/path/to/pdfs --workers 3   # process a folder
uv run python batch.py --aggregate-only             # rebuild tables from the OUTER cache
uv run python batch.py --include-flagged            # include needs_review reports
```

After a **validator** change: delete `data/cache/*.json` and run the pipeline
normally — extraction is reused from `raw/`, so re-validation is free.
`--aggregate-only` will NOT re-validate; on an empty outer cache it has nothing
to load. After an **extraction prompt** change, delete the matching `raw/` file too.

**Outputs** (`data/aggregate/`): `properties.csv`, `systems.csv`,
`components.csv`, `manifest.csv`, matching `.parquet`, and `s3/` partitioned by
`firm=` / `state=` (leave-one-firm-out and leave-one-region-out CV become a
prefix filter).

---
## Validation stack
0. **Extract** — per property field: `value + page + snippet + confidence`.
1. **Deterministic** — types, ranges, categories, `RUL <= EUL`.
2. **Coercion + normalization** — numerics forced to numbers, ranges split into min/max, firm category spellings folded onto the canonical scale.
3. **Completeness** — silently-empty tables, and tables whose rows carry no totals. Reconciliation cannot see these.
4. **Reconciliation** — extracted line items must sum to the report's own printed totals at all three layers. **Strongest check.**
5. **Grounding** — cited snippet must appear in the PDF (tolerant of extractor differences, strict about content). Advisory: does not block.
6. **Arithmetic** *(new, free)* — a row's own numbers must agree:
   `quantity x unit_cost` vs the stated extended cost, and `EUL - age == RUL`.
   Catches transcription errors inside a row that sit inside a correct total,
   which reconciliation cannot see. Advisory. 123 hits.
7. **Number presence** *(new, free)* — each headline dollar figure must appear
   somewhere in the PDF text layer, searched as a human would see it written.
   Tests the VALUE, where grounding tests the model's own chosen snippet.
   Advisory. 14 hits.
8. **Judge** — second model re-reads the PDF, only for fields that failed the
   layers above or came back low-confidence. Fails soft. **~a third of API
   spend; `--no-judge` disables it and layers 1-7 still run.**

Blocking policy lives in `batch.py`, not `validate.py`.

---
## Cross-firm findings (the substantive result)
- **EBI / EMG / Bureau Veritas** share a table shape: EUL, EFF AGE, RUL, Quantity, Unit, Unit Cost, Cycle Replace, Replace Percent, Year 1–12, Total.
- **Partner Engineering** has no replace-percent column — uses "On Site Qty" + "Qty in Eval Period" (`qty_in_eval_period`).
- **Partner has no condition summary table at all.** Conditions appear only in narrative prose; the extractor derives one systems row per numbered section (Mariella: 0 → 37 rows).
- **Partner's condition vocabulary is its own** — "not applicable", "functional". Synonyms fold; "functional" does not.
- **EMG and BV split Table 1** into separate Immediate and Short Term budgets with separate stated totals (`table="short_term"`).
- **Bureau Veritas puts cost-table content near the back** (Wickliffe pp. 169–172), contradicting the front-loading assumption the pipeline was built on.
- **Senior housing is measured in beds, not units.** Two of four reports are assisted living / memory care. Added `num_beds`, `care_types`.

### Added 30 Aug, from the full corpus
- **Gabion** ("Capital Considerations") shares almost nothing with the others: no EUL / EFF AGE / RUL column at all, immediate vs reserve marked by a row class prefix (`I.N.` / `R.R.`), calendar-year columns, and inflated dollars in them.
- **Terracon** runs `R - 1, R - 2 …` numbered rows with EUL only — no eff age, no RUL — and quotes EUL as a range ("5-7", "20-25").
- **AEI's Homewood Suites set** is a 5-year term with `Overall Expected Life` and `Remaining Life` as ranges, a footnote column, and a two-line header the text layer breaks apart ("Overall Expected Remaining … Life (a) Life (b)").
- **Tetra Tech** calls effective age "Reflective Age" and writes "Varies" in the life columns on about a quarter of its rows.
- **LandScience and CBC have no component table at all.** Costs appear as numbered items inside a section-level table with "Immed. Cost" / "Reserve Cost" columns, or as a numbered "Immediate Need Repairs Estimate" list. Component rows must be built from those items — an empty `components` array is the wrong answer, not the honest one.
- **EBI, Atwell and Metropolitan Solutions** cost per section inline ("COST SUMMARY / Recommendation / EUL / EFF AGE / RUL / Year / Cost") rather than in one consolidated table, and use "Immed" where a term year number would go.
- **Two reports cannot be read from their text layer.** 7800 Alpha Road's is a repeated watermark ("Confidentially provided to …") over what is effectively a scan — zero currency amounts in 93 pages. Westcott Apartments has a broken font encoding (its cover reads `)5('',(0$&` where it should read FREDDIEMAC). The model still sees the pages as images, so extraction can work; **grounding cannot verify anything on these two**, and page selection falls back to the front block.
- **Harbor at Harmony Crossing is not a reserve-table report.** It is a pre-purchase bundle with vendor invoices and work orders attached. It will legitimately produce few or no component rows; completeness will flag it, and that flag is correct.

**The bug reconciliation caught:** Mission Springs' immediate repairs summed to
$142,300 against a stated $12,500 — EMG's short-term repairs ($129,800) were
being folded into the immediate bucket. Without reconciliation this would have
silently corrupted the cost labels. Good concrete example for Kristen.

---
## Open items

**The owner is currently working through: are the CSVs good enough to train
on?** Everything below is ordered against that question.

1. **Decide on a final re-extraction under a frozen prompt.** The 134
   extractions span ~6 prompt versions and the quality correlates with firm
   (see "The thing that most needs fixing" above). This is the biggest threat
   to the leave-one-firm-out claim. `--no-judge` puts it at $64; a Haiku 4.5
   trial on 10 reports (~$2) would tell you whether $32 is achievable, using
   reconciliation pass rate against the Sonnet results already on disk as the
   metric. **Note Haiku's context is 200K and the largest request in this
   corpus is 191,326 tokens - 96% of the limit, no headroom.**
2. **Normalise `description`** — `taxonomy.py` now does 92.2%; re-fit the tail
   on the full corpus and decide the category granularity for modelling.
3. **Decide the missing-lives policy.** 1,336 of 2,409 rows have EUL+age+RUL,
   and the missingness is firm-correlated, not random.
4. **Re-extract `7800 Alpha Road`** (~$0.60) — the only corpus report with no
   extraction. Expect a thin result; its text layer is a watermark.
5. **Investigate the ~10 under-extracted reports** before paying to re-run
   them. `scratchpad/gap.py`-style money-density scanning is free and
   distinguishes "genuinely sparse" from "we missed the table".
6. **Build the gold set** (10-15 hand-labelled reports). There is still **no
   measured extraction accuracy**, so the noise floor in the training data is
   unknown. Reconciliation proves totals tie out; it cannot prove a
   description was transcribed correctly or a quantity read off the right row.
   This is the number Kristen will ask for.
7. **5 reports still have an unresolved firm** — `Partners`,
   `Property Solutions Inc.`, `Partner Assessment Corp`. The page-scan found
   no registered firm in those PDFs. Firm is the CV partition key.
8. **Decide what the non-ordinal conditions mean.** `CONDITION_ORDINAL` ranks
   only excellent/good/fair/poor. Everything else - functional, adequate,
   average, well maintained, non-compliant, dated, no issues observed, and the
   numeric 1-5 ratings - is carried unranked, which means `condition` is
   currently unusable as a numeric feature on any report using them.
9. **Label join.** These are engineer ESTIMATES, not realised CapEx. A cost
   model trained on `properties.csv` predicts what an assessor wrote, not what
   was spent. True labels need Jimmy's spend records joined on property, plus
   right-censoring logic.
10. **Prompt caching is DONE and working** (call A writes, call B reads at
    0.1x). The open piece is the 5-minute TTL expiring before a late judge
    call, which cost ~36% extra on slow reports. Moot while `--no-judge` is in
    use.
11. **Extended thinking is ON and is most of the output bill.** `EFFORT` is
    now `"medium"` in `extract.py` (shared by `judge.py`) - see the note there
    for why. It cut thinking ~80% on call A, made runs faster, and kept the
    canary clean.
12. **Tell the teammate Textract is redundant** — output is CSV/S3, not
    DynamoDB.

## Gotchas
- **Downloaded files must actually be copied into the repo.** Verify with a `grep -c "<new marker>" <file>` before assuming a fix ran.
- **Two-level cache.** `data/cache/raw/` = expensive extractions; `data/cache/*.json` = validated results. Validator change → delete outer only. Prompt change → delete the matching `raw/` file too.
- **`data/aggregate/` is disposable.** `data/cache/` is not.
- **zsh errors on globs that match nothing** and stops the command line.
- **Only reports that pass validation enter the aggregate.** If a meaningful fraction of 132 fails, that is a sample-bias question, not just a QA one. With 16 firms in the corpus, check *which* firms fail before treating a failure rate as a QA number — a validator that rejects one firm's layout wholesale looks the same as a 12% error rate and means something completely different.
- **The inbox has 138 files and 132 reports.** `batch.py` dedupes by default. If you list PDFs yourself, dedupe yourself, or you will pay twice and correlate the training set.
- **`corpus_profile.csv` is regenerable and free.** Re-run the profiler after any change to `_TABLE_MARKERS`, `_structural_score` or `FIRM_PATTERNS` — it imports them directly, so the profile predicts what the pipeline will really send.

---
- **The 32MB API limit is on the REQUEST, and the PDF is base64 there** - 4/3
  the size on disk. `MAX_PDF_BYTES` is 22MB of PDF (~29.3MB encoded) for that
  reason. Checking raw bytes against 30MB sent a ~42MB request and got a 413
  after the whole page-selection pass had run.
- **`--aggregate-only` reads the outer cache, it does NOT re-validate.** After
  any validator change run `verify_offline.py` first, or the CSVs will carry
  verdicts from an older ruleset.
- **Deleting `data/cache/*.json` and re-running re-fires the judge on every
  report.** That is the expensive path. Use `verify_offline.py` instead.
- **A finished run can still lose all its output.** 132 extractions completed
  and then the parquet write died on a mixed-type column, producing nothing.
  Fields in `PROPERTY_FIELDS` with no `PROPERTY_META` entry are never coerced
  and reach pandas as object columns - `num_buildings` did exactly this. The
  S3 partition writer now falls back to CSV per partition.
- **`_dedupe` reads every file to hash it**, so one unreadable path used to
  abort a 130-report run at report zero. It now skips unreadable files loudly.
- **Only ONE of the four original reports is in the inbox.** Wickliffe,
  Mission Springs and Mariella have no PDF in the repo, only cached
  extractions - do not delete `data/cache/raw/` for those three, there is
  nothing to regenerate them from.
- **Check the account's API usage limit before a long run**, not just the
  dollar cost. Batch 2 died at report 55 on a usage cap, and a later pass
  exhausted credits mid-flight. Errors are not cached, so nothing was lost -
  but re-running blind wastes hours.

## Environment
- `uv` for env management, Cursor as editor, macOS
- `.env`: `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=pca-extract`
- LangSmith project `pca-extract` (Personal workspace) — traces show `pca_pipeline` with nested `extract_pca` / `judge_fields` spans
- Model string `claude-sonnet-5` in `extract.py` and `judge.py` — verified 29 Aug, returns `['text']`, no default thinking
- `MAX_PDF_PAGES = 90` in `schema.py`; API caps PDF requests at 100 pages / 32MB
- `RESERVE_YEARS = 15` in `schema.py` — reserve terms in this corpus run 5 to 15 years; `year_*` column names derive from it
- Corpus lives in `data/inbox/PCA Reports/` (138 files, 132 unique)
