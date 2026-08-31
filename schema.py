"""
Schema built from the actual EBI Consulting PCR structure (ASTM E 2018-15).
Three record types per report:

  PROPERTY   one row  — identity, metadata, summary totals
  SYSTEMS    ~20 rows — one per numbered section (2.1..5.2): condition + costs
  COMPONENTS many rows — Table 1 (immediate) + Table 2 (reserves) line items,
                         including EUL / effective age / RUL per component.
                         This is the timing layer's training data.

Other firms will phrase and organize differently; the extractor maps their
wording onto THESE names. Extend the lists as new firms' reports reveal
fields — add, don't rename, so downstream joins stay stable.

Widened after reading the full 132-report corpus: 16 firms and 9 distinct
cost-table layouts, not 4 and 4. See FIRM_PATTERNS and the shape catalogue
further down for what that cost in schema terms.
"""

# ── CONDITION VOCABULARY ───────────────────────────────────────────────────
# Two lists, deliberately. CONDITIONS is everything a firm is allowed to say;
# CONDITION_ORDINAL is the much smaller set whose position on the wear axis is
# unambiguous across firms. A rating outside CONDITIONS fails the category
# check and takes the whole report to needs_review, so this list has to be
# generous - but a rating that lands in CONDITION_ORDINAL becomes a NUMBER in
# the timing model, so that one has to be strict. Adding a word here is cheap;
# ranking it wrongly is a silent modelling error.
#
# The ordinal four are the only ones every rating legend in the corpus agrees
# on. Everything below them is carried verbatim and ranked by nobody.
CONDITION_ORDINAL = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}

# Ratings that assert the thing WORKS without placing it on the wear axis.
# Partner Engineering's "functional" was the first; the wider corpus adds
# "adequate", "acceptable", "serviceable" and "satisfactory", which are the
# same kind of statement in different words. Mapping any of them onto
# good/fair would invent precision the report does not contain.
CONDITION_FUNCTIONAL = ["functional", "adequate", "acceptable",
                        "serviceable", "satisfactory", "operational"]

# Ratings the corpus uses whose rank is firm-dependent, so they cannot be
# ranked at all:
#
#   "average" - The Carlisle Naples prints a 1-5 legend where it is its OWN
#     level BETWEEN good and fair ("3 - Average: Normal condition for the age
#     of components"). CBC, Nova and Atwell print a 4-level legend that
#     defines GOOD as "average to above-average condition". The same word sits
#     at two different positions depending on who wrote the report, which is
#     exactly why it must not get a number. Same for its neighbours.
#   "marginal" / "deficient" / "critical" - distress words no legend in the
#     corpus defines. They plainly mean "bad", but "how bad" is not stated,
#     and poor already occupies the bottom of the ranked scale.
#
# Gabion adds a second kind of unrankable rating: statements about CODE
# COMPLIANCE and AGE rather than wear. A "dated" kitchen can be in perfectly
# good condition, and "non-compliant" describes a code finding, not how worn
# the thing is - neither belongs anywhere on an excellent-to-poor axis. They
# are real findings a report made, so they are carried, not discarded.
#
# Batch 1 added a third kind: statements that a defect was LOOKED FOR AND NOT
# FOUND ("no issues observed", "no evidence of movement"), and ADA/code
# accessibility findings ("compliant", "partially accessible"). Neither is a
# wear rating. "No issues observed" is genuinely weaker than "good" - it says
# the assessor saw nothing wrong, not that the thing is in good shape - so it
# is carried, not folded.
CONDITION_UNRANKED = ["average", "above average", "below average",
                      "marginal", "deficient", "critical",
                      "non-compliant", "dated", "obsolete",
                      "no issues observed", "compliant",
                      "partially accessible", "generally accessible",
                      "worn", "inefficient", "inadequate", "current",
                      # Maintenance-practice and trend statements. "Well
                      # maintained" describes upkeep, not wear - a well
                      # maintained 30-year roof is still a 30-year roof - and
                      # "declining"/"fading" give a direction of travel rather
                      # than a position on the scale.
                      "well maintained", "typical", "consistent",
                      "level and stable", "sufficient",
                      "declining", "fading"]

CONDITIONS = (list(CONDITION_ORDINAL) + ["na"]
              + CONDITION_FUNCTIONAL + CONDITION_UNRANKED)

# Firms spell the same rating differently. These are exact synonyms only -
# nothing here changes the meaning of a rating, it only rewrites how a firm
# spells one. Anything genuinely ambiguous belongs in CONDITIONS, not here.
CONDITION_SYNONYMS = {
    "n/a": "na",
    "n.a.": "na",
    "n/a.": "na",
    "not applicable": "na",
    "does not apply": "na",
    "not assessed": "na",
    "not observed": "na",
    "not rated": "na",
    "not present": "na",
    # "The assessment did not produce a rating" - the same statement as
    # "not assessed", which already folds here.
    "not performed": "na",
    "not evaluated": "na",
    "unknown": "na",
    "outdated": "dated",
    "operating condition": "operational",
    "operable": "operational",
    # The feature is absent, which is the "none" case already folded above.
    "none in place": "na",
    "not in place": "na",
    "substantially compliant": "compliant",
    "fully compliant": "compliant",
    "not provided": "na",
    "not present at property": "na",
    "none - not installed": "na",
    "not installed": "na",
    # "We looked and found nothing wrong" - one statement, many spellings.
    "no evidence observed": "no issues observed",
    "none observed": "no issues observed",
    # "It differs across the property" - no single rating exists.
    "varying": "na",
    "varies": "na",
    "mixed": "na",
    "no defects observed": "no issues observed",
    "no evidence of movement": "no issues observed",
    "no significant issues": "no issues observed",
    "no issues noted": "no issues observed",
    "no deficiencies observed": "no issues observed",
    "no visible defects": "no issues observed",
    "none": "na",
    "excellent condition": "excellent",
    "good condition": "good",
    "fair condition": "fair",
    "poor condition": "poor",
    "very good": "good",
    "very poor": "poor",
    # "New" is not a fifth rating - it is how most legends in this corpus
    # SPELL excellent. Verbatim, from the reports themselves: "Excellent: New
    # or like New" (CBC, Nova, Atwell, LandScience, Gabion) and "1 - Excellent
    # New or like new condition" (The Carlisle Naples). Folding it is reading
    # the firm's own definition, not a judgement call.
    "new": "excellent",
    "like new": "excellent",
    "new or like new": "excellent",
    "as new": "excellent",
}
# Some firms tick more than one condition box for a system (e.g. "fair; poor").
# condition_secondary may therefore hold a semicolon-joined value; the
# validator splits on ";" and checks each part.

# ── PROPERTY (one row per report) ──────────────────────────────────────────
PROPERTY_FIELDS = [
    "property_id", "source_file",          # join keys, filled by the pipeline
    "property_name", "property_address", "city", "state", "zip",
    "report_firm", "project_number", "client_name",
    "report_date", "assessment_date",
    "property_type", "year_built", "renovation_years",     # e.g. "1985; 2018"
    "num_stories", "num_stories_min", "num_stories_max",
    "num_units", "num_residential_units", "num_commercial_units",
    "num_beds", "care_types",              # senior housing: beds, not units
    "num_rooms",                           # hotels: keys, not units
    "unit_basis",                          # which denominator the per-unit
                                           # metrics below are actually per
    "building_age_years",                  # some firms state this directly
    "net_rentable_sf", "site_acres", "num_parcels", "num_buildings",
    "basement",                                            # e.g. "full, unfinished"
    "construction_type", "facade_materials", "roof_types",
    "heating_source", "cooling_type", "water_heater_type",
    "num_elevators", "fire_sprinklers", "emergency_generator",
    "overall_condition", "overall_rul_years",
    "immediate_repairs_total_usd", "immediate_repairs_per_unit_usd",
    "short_term_repairs_total_usd",        # EMG/BV split this out from immediate
    "non_critical_repairs_total_usd",      # Tetra Tech splits immediate needs by
                                           # SEVERITY, not timing - see the
                                           # "non_critical" note in COMPONENT_FIELDS
    "priority_repairs_total_usd",          # Freddie-Mac-style reports (Villa
    "operational_repairs_total_usd",       # Oaks) use a third axis again:
                                           # Immediate / Priority / Operational
                                           # / Capital, each with its own
                                           # stated total. THIS LIST WILL KEEP
                                           # GROWING - every firm slices the
                                           # near-term bucket its own way, and
                                           # folding them together is what
                                           # breaks the stated-total tie-out.
    "reserves_total_uninflated_usd", "reserves_total_inflated_usd",
    "reserves_total_present_value_usd",    # Gabion publishes a DISCOUNTED total
                                           # ("Total Present Value (With
                                           # Contingency)") and no undiscounted
                                           # one. It is not a sum of line items
                                           # and must never be reconciled as
                                           # though it were - but it is a real
                                           # number the report states, so it
                                           # gets its own column instead of
                                           # being forced into one that lies.
    "reserves_per_unit_usd", "inflation_rate_pct", "reserve_term_years",
    "contingency_pct", "contingency_usd",  # a markup ADDED ON TOP of the line
                                           # items, so the items will never sum
                                           # to the stated total without it -
                                           # see RECONCILE note below
    "reserve_per_sf_year_uninflated", "reserve_per_sf_year_inflated",
    "reserve_per_unit_year_uninflated", "reserve_per_unit_year_inflated",
]

PROPERTY_META = {
    "report_date":        {"type": "date"},
    "assessment_date":    {"type": "date"},
    "year_built":         {"type": "number", "min": 1850, "max": 2026},
    "num_stories":        {"type": "number", "min": 1, "max": 120},
    "num_stories_min":    {"type": "number", "min": 1, "max": 120},
    "num_stories_max":    {"type": "number", "min": 1, "max": 120},
    "num_units":          {"type": "number", "min": 1, "max": 5000},
    "num_residential_units": {"type": "number", "min": 0, "max": 5000},
    "num_commercial_units":  {"type": "number", "min": 0, "max": 1000},
    "net_rentable_sf":    {"type": "number", "min": 1000, "max": 10_000_000},
    "site_acres":         {"type": "number", "min": 0.01, "max": 2000},
    "num_parcels":        {"type": "number", "min": 1, "max": 100},
    "num_elevators":      {"type": "number", "min": 0, "max": 60},
    "num_beds":           {"type": "number", "min": 0, "max": 2000},
    "num_rooms":          {"type": "number", "min": 0, "max": 5000},
    "unit_basis":         {"type": "category",
                           "allowed": ["units", "beds", "rooms", "sf"]},
    "building_age_years": {"type": "number", "min": 0, "max": 200},
    "overall_condition":  {"type": "category", "allowed": CONDITIONS},
    "overall_rul_years":  {"type": "number", "min": 0, "max": 100},
    "immediate_repairs_total_usd":    {"type": "number", "min": 0, "max": 150_000_000},
    "short_term_repairs_total_usd":   {"type": "number", "min": 0, "max": 150_000_000},
    "non_critical_repairs_total_usd": {"type": "number", "min": 0, "max": 150_000_000},
    "priority_repairs_total_usd":     {"type": "number", "min": 0, "max": 150_000_000},
    "operational_repairs_total_usd":  {"type": "number", "min": 0, "max": 150_000_000},
    "reserves_total_present_value_usd": {"type": "number", "min": 0, "max": 200_000_000},
    "reserves_total_uninflated_usd":  {"type": "number", "min": 0, "max": 150_000_000},
    "reserves_total_inflated_usd":    {"type": "number", "min": 0, "max": 200_000_000},
    "inflation_rate_pct": {"type": "number", "min": 0, "max": 15},
    # These were in PROPERTY_FIELDS but had no meta, so coerce_types never
    # touched them and they reached the CSV as whatever the model emitted -
    # "2" on one report, 2 on the next. Pandas then held an object column and
    # the Parquet write died with ArrowTypeError AFTER the whole 132-report
    # run had completed. Typing them is the actual fix; the guard in
    # batch.write_outputs is the seatbelt.
    "num_buildings":      {"type": "number", "min": 1, "max": 500},
    "reserves_per_unit_usd":          {"type": "number", "min": 0, "max": 1_000_000},
    "immediate_repairs_per_unit_usd": {"type": "number", "min": 0, "max": 1_000_000},
    "reserve_per_sf_year_uninflated": {"type": "number", "min": 0, "max": 1_000},
    "reserve_per_sf_year_inflated":   {"type": "number", "min": 0, "max": 1_000},
    "reserve_per_unit_year_uninflated": {"type": "number", "min": 0, "max": 100_000},
    "reserve_per_unit_year_inflated":   {"type": "number", "min": 0, "max": 100_000},
    "contingency_pct":    {"type": "number", "min": 0, "max": 50},
    "contingency_usd":    {"type": "number", "min": 0, "max": 50_000_000},
    "reserve_term_years": {"type": "number", "min": 1, "max": 30},
}

# Numeric property fields whose source text is legitimately a RANGE rather than
# a single value. Real examples: "1-4" (Bureau Veritas, Wickliffe) and "One and
# two floors" (EMG, Mission Springs) - two buildings of different heights on one
# parcel, which is a fact about the property, not a bad extraction. Nulling
# these threw away the field AND excluded the whole report over it.
# field -> (min_field, max_field). The base field gets the max, so anything
# already keyed on num_stories keeps working.
RANGE_NUMBER_FIELDS = {
    "num_stories": ("num_stories_min", "num_stories_max"),
}

# ── SYSTEMS (one row per numbered section) ─────────────────────────────────
# From the Executive Summary Table: 2.1 Topography .. 5.2 Fire Department.
SYSTEM_FIELDS = [
    "property_id", "report_firm",          # join keys, filled by the pipeline
    "section_code",            # "3.4"
    "system_name",             # "Roofing"
    "condition",               # primary rating (lowercase)
    "condition_secondary",     # second X when rated e.g. "Good to Fair", else null
    "condition_rating_numeric",# firms that rate on a 1-5 scale (EMG) instead of
                               # words. The bare number is NOT translated into a
                               # word: The Carlisle Naples prints a legend where
                               # 1=Excellent..5=Poor, but the EMG reports in this
                               # corpus print no legend at all, and a scale whose
                               # direction is assumed is worse than one left
                               # numeric. Kept as a number, mapped only when a
                               # report states its own legend.
    "action_required",         # verbatim-ish: "Replace", "Refurbish, Repair", "None"
    "immediate_repairs_usd",   # from the exec summary row; null if blank
    "short_term_repairs_usd",  # firms that split the near-term bucket by
    "non_critical_repairs_usd",# TIMING and by SEVERITY respectively - the
                               # same split the components layer carries.
                               # Without these, a system row folds every
                               # near-term cost into immediate_repairs_usd and
                               # the systems sum overshoots the stated
                               # immediate total. Measured on Tetra Tech's
                               # Maybelle Carter: systems summed 36,175
                               # against a stated 3,750, which is exactly
                               # 3,750 critical + 32,425 non-critical.
    "replacement_reserves_usd",
]

SYSTEM_META = {
    "condition":            {"type": "category", "allowed": CONDITIONS},
    "condition_secondary":  {"type": "category", "allowed": CONDITIONS},
    "condition_rating_numeric": {"type": "number", "min": 1, "max": 5},
    "immediate_repairs_usd":    {"type": "number", "min": 0, "max": 50_000_000},
    "short_term_repairs_usd":   {"type": "number", "min": 0, "max": 50_000_000},
    "non_critical_repairs_usd": {"type": "number", "min": 0, "max": 50_000_000},
    "replacement_reserves_usd": {"type": "number", "min": 0, "max": 50_000_000},
}

# ── COMPONENTS (one row per Table 1 / Table 2 line item) ───────────────────
# Longest reserve term in the corpus. Most firms run 12 years; NV5, Terracon,
# Tetra Tech, Lender Consulting and Gabion run 10; AEI's Homewood Suites set
# runs 5; Partner's German Church report runs 15, which is where this number
# comes from. Emitting only year_1..year_12 silently dropped that report's
# last three years - and because the schedule then no longer summed to the
# row's own total, it looked like a bad extraction rather than a short schema.
# Derive the column names from this constant; do not hand-write them.
RESERVE_YEARS = 15
YEAR_FIELDS = [f"year_{i}" for i in range(1, RESERVE_YEARS + 1)]

COMPONENT_FIELDS = [
    "property_id", "report_firm",          # join keys, filled by the pipeline
    "table",                # "immediate" | "short_term" | "non_critical" |
                            # "reserve". These are separate budgets and must
                            # never be summed together, or the stated totals
                            # will not tie out.
                            #
                            # Firms split the near-term bucket on two different
                            # axes, and both splits have to survive:
                            #   TIMING   - EMG and Bureau Veritas run an
                            #     Immediate column and a Short Term column.
                            #   SEVERITY - Tetra Tech divides immediate needs
                            #     into "Immediate Critical Repair Needs"
                            #     (health and safety) and "Immediate
                            #     Non-critical Repair Needs", in two separate
                            #     numbered sections with separate totals.
                            # Folding non_critical into short_term would assert
                            # these mean the same thing. They do not: one is
                            # about WHEN the money is spent, the other about
                            # WHY, and an underwriter treats a critical
                            # life-safety repair differently from a cosmetic
                            # one due at the same time.
    "section_code",         # "3.4"
    "description",          # "EPDM roof replacement"
    "eul_years",            # Table 2 only; null on Table 1 rows
    "eul_years_min", "eul_years_max",      # see RANGE_COMPONENT_FIELDS
    "effective_age_years",  # null when report says "var"/"Varies"
    "rul_years",            # null when "var"/"Varies"
    "rul_years_min", "rul_years_max",
    "rul_varies",           # true when the report said var/varies
    "quantity", "unit",     # 18800, "SF" | "ALW" | "EA" | "LF" | "UNIT" | "LS" | "TON"
    "qty_in_eval_period",   # Partner only: on-site qty x cycles in term
    "unit_cost_usd",
    "cycle_replace_cost_usd",
    "replace_percent",      # % of one cycle cost incurred over the term.
                            # <100 = partial (EMG 60%); 100 = once;
                            # 200/300 = multiple cycles; 1200 = annual.
                            # Null for firms that use qty_in_eval_period instead.
    "start_year",           # first term year the spend occurs in. Gabion
                            # states it outright ("Start Year of Occurrence");
                            # for everyone else it is the first non-empty
                            # year column, which is not the same fact and is
                            # therefore left null rather than inferred.
    "cycle_years",          # Gabion "Work Span / Cycle": years between
                            # recurrences. The recurrence fact that
                            # replace_percent encodes for shape-A firms.
    "total_cost_usd",
    "years_inflated",       # TRUE when the year columns carry inflated
                            # dollars. Gabion's do - a $51,500 item shows
                            # $65,239 in a later year - so their schedule
                            # will NOT sum to an uninflated total, and
                            # reconciliation must not read that as a
                            # miscount. Most firms publish uninflated year
                            # columns with a separate inflated total row.
    # Reserve spend schedule (null on Table 1 rows and on empty years).
] + YEAR_FIELDS

COMPONENT_META = {
    "table":               {"type": "category",
                            "allowed": ["immediate", "short_term",
                                        "non_critical", "priority",
                                        "operational", "critical",
                                        "deferred", "accessibility",
                                        "life_safety", "reserve"]},
    "eul_years":           {"type": "number", "min": 0, "max": 100},
    "effective_age_years": {"type": "number", "min": 0, "max": 100},
    # NEGATIVE RUL IS VALID AND MEANINGFUL. Atwell computes RUL = EUL - effective
    # age and lets it go negative: a component with a 15-year EUL and a 20-year
    # effective age gets rul_years = -5, meaning it is five years PAST its
    # expected life. That is a deferred-maintenance signal and one of the more
    # informative things in the table - clamping it to 0 or nulling it would
    # erase exactly the overdue components the timing model most needs to see.
    # Two reports (Lakewood, Eagle Creek) were rejected outright over this.
    "rul_years":           {"type": "number", "min": -60, "max": 100},
    "quantity":            {"type": "number", "min": 0, "max": 10_000_000},
    "qty_in_eval_period":  {"type": "number", "min": 0, "max": 50_000_000},
    "unit_cost_usd":       {"type": "number", "min": 0, "max": 10_000_000},
    "cycle_replace_cost_usd": {"type": "number", "min": 0, "max": 50_000_000},
    "replace_percent":     {"type": "number", "min": 0, "max": 5000},
    "total_cost_usd":      {"type": "number", "min": 0, "max": 50_000_000},
    "eul_years_min":       {"type": "number", "min": 0, "max": 100},
    "eul_years_max":       {"type": "number", "min": 0, "max": 100},
    "rul_years_min":       {"type": "number", "min": 0, "max": 100},
    "rul_years_max":       {"type": "number", "min": 0, "max": 100},
    "start_year":          {"type": "number", "min": 0, "max": RESERVE_YEARS},
    "cycle_years":         {"type": "number", "min": 0, "max": 100},
    **{y: {"type": "number", "min": 0, "max": 50_000_000} for y in YEAR_FIELDS},
}

# Component lives are quoted as RANGES by several firms, and nulling them
# throws away the timing layer's whole reason for existing. Real examples:
# AEI writes an expected life of "10-15" with a remaining life of "1-2"
# (Homewood Suites HP set); Terracon writes "5-7", "15-20", "20-25" (ARIUM).
# Same treatment as num_stories: split into min/max, base field takes the MAX,
# so anything already keyed on eul_years keeps working and the uncertainty is
# still on the record instead of being rounded away.
# Every `table` value except "reserve" is money to be spent in the near term;
# firms just disagree about how to slice it (timing, severity, urgency, cause).
# Nine buckets is already a lot and the list will keep growing, so downstream
# gets a stable two-value axis derived from it rather than having to know every
# firm's vocabulary. Reconciliation also uses this: when a report states one
# combined immediate total but itemises into several near-term buckets, the
# sum of ALL of them is the correct comparison.
NEAR_TERM_TABLES = ("immediate", "short_term", "non_critical", "priority",
                    "operational", "critical", "deferred", "accessibility",
                    "life_safety")


def table_horizon(table) -> str:
    """-> "near_term" | "reserve" | None. The stable grouping for modelling."""
    if table in NEAR_TERM_TABLES:
        return "near_term"
    return "reserve" if table == "reserve" else None


RANGE_COMPONENT_FIELDS = {
    "eul_years": ("eul_years_min", "eul_years_max"),
    "rul_years": ("rul_years_min", "rul_years_max"),
}

# Category fields to normalise before checking, by record layer. Used by
# validate.coerce_types to rewrite firm spellings onto the canonical scale.
UNIT_BASIS_SYNONYMS = {
    "unit": "units", "dwelling units": "units", "apartments": "units",
    "apartment units": "units", "residential units": "units",
    "bed": "beds", "beds/units": "beds",
    "room": "rooms", "key": "rooms", "keys": "rooms", "guestrooms": "rooms",
    "guest rooms": "rooms", "per key": "rooms",
    "square feet": "sf", "square foot": "sf", "gsf": "sf", "nrsf": "sf",
}

CATEGORY_SYNONYM_FIELDS = {
    "property":   {"overall_condition": CONDITION_SYNONYMS,
                   "unit_basis": UNIT_BASIS_SYNONYMS},
    "systems":    {"condition": CONDITION_SYNONYMS,
                   "condition_secondary": CONDITION_SYNONYMS},
    "components": {},
}

# US state -> two-letter code. `state` is the S3 partition key for
# leave-one-REGION-out CV, and reports write it both ways: the corpus came
# back with "Florida" 23 times and "FL" 17 times, "Texas" 25 and "TX" 5 -
# 33 distinct values for ~25 states. Left unnormalised, Florida is two
# regions and a held-out Florida fold still has Florida in training. Same
# class of leak as the firm-name split, and just as invisible.
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington, d.c.": "DC",
}

# ── FIRMS AND TABLE SHAPES ─────────────────────────────────────────────────
# One registry, imported by extract.py and profileCorpus.py, so "which firms
# do we handle" has a single answer. Ordered: the first pattern to match the
# cover page wins, so put specific names before generic abbreviations.
#
# Counts are from the 138-file inbox (132 unique reports; see DUPLICATES in
# batch.py). Firms are listed whether or not their table shape is handled -
# an unhandled report should be NAMED in the manifest, not lumped into
# UNKNOWN where it cannot be counted.
FIRM_PATTERNS = [
    (r"ebi consulting|\bebi\b",                     "EBI Consulting"),
    (r"\bemg\b|emgcorp",                            "EMG"),
    (r"bureau veritas|\bbv project\b",              "Bureau Veritas"),
    # "Partner Assessment Corporation" is a DIFFERENT company from Partner
    # Engineering and Science - matching on the bare word "partner" merged
    # them, which is the same leave-one-firm-out leak as splitting one firm in
    # two, running the other way. Both are listed explicitly.
    (r"partner engineering|partner esi|partner project", "Partner Engineering"),
    (r"partner assessment",                         "Partner Assessment Corp"),
    (r"faithful\+?\s*gould",                        "Faithful+Gould"),
    (r"\bgrs group\b",                              "GRS Group"),
    (r"\bcbre\b",                                   "CBRE"),
    (r"\bf3,? inc",                                  "f3"),
    # Verified against their cover pages ("Prepared by"), not guessed. All
    # three were sitting in the firm column looking like extraction errors
    # and are in fact correct - "Partners" really is a firm in Solon, Ohio,
    # and CSI is how Consulting Solutions Inc stamps its project numbers.
    # Property Solutions Inc. was briefly listed as CLIENT noise by mistake,
    # which would have thrown away a correct attribution.
    (r"property solutions inc",                     "Property Solutions Inc"),
    (r"consulting solutions inc|\bcsi project\b",   "Consulting Solutions Inc"),
    (r"\bpartners\b(?!\s+(?:in|of|group\b))",       "Partners"),
    (r"aei consultants|\baei\b",                    "AEI"),
    (r"nova consulting|\bnova\b",                   "Nova"),
    (r"landscience",                                "LandScience"),
    (r"commercial building consultants|\bcbc\b",     "CBC"),
    (r"gabion|\bgrea-\d",                           "Gabion"),
    (r"tetra tech",                                 "Tetra Tech"),
    (r"terracon",                                   "Terracon"),
    (r"metropolitan solutions",                     "Metropolitan Solutions"),
    (r"atwell",                                     "Atwell"),
    (r"\bepic\b",                                   "EPIC"),
    (r"\bnv5\b",                                    "NV5"),
    (r"lender consulting",                          "Lender Consulting"),
    (r"varian associates",                          "Varian Associates"),
    (r"sierra piedmont",                            "Sierra Piedmont"),
    (r"\bbbg\b",                                    "BBG"),
    (r"merritt ?(?:and|&) ?harris|jll",             "JLL / Merritt & Harris"),
    (r"criterium|hillmann|\becs\b|aecom|dominion|\bwsp\b|arcadis",
                                                    "OTHER"),
]

# Owners, lenders and investors that appear prominently on these covers under
# "Prepared for" / "Confidentially provided to". They are NOT assessing firms,
# and the distinction is invisible downstream: both are plausible company
# names, so a swapped one enters the training tables as a valid-looking firm
# and silently splits or merges a leave-one-firm-out fold.
#
# It has happened. In the first seven-report run an AEI report came back as
# "Bridge House Advisors Corp." and an EPIC report as "Partners". Nothing
# flagged either. validate._client_named_as_firm turns that into a property
# issue so the judge re-reads the cover instead of the value standing.
#
# Note Gabion is deliberately NOT here. It reads like a fund name and was
# treated as client noise at first, but Gabion Real Estate Advisors is the
# assessing firm on four reports in this corpus.
CLIENT_NAMES = (
    r"lloyd jones|realinsight|argentic|bridge house|hps investment|"
    r"midland loan|berkeley point|skyline investments|lodging capital|"
    r"mk associates|property solutions inc|slg consulting|renasant bank|"
    r"inspired healthcare|berkadia|jones lang lasalle|\bjll multifamily\b|"
    r"finlay management|ten-x|tetra tech capital"
)

# The distinct cost-table LAYOUTS behind those firms. Firms matter less than
# shapes - several "new" firms turn out to emit a table already handled, and
# the prompt is written against shapes, not names. Documented here because
# this is the list that decides whether a new report needs code or just runs.
#
#   A  replace-percent  EUL / EFF AGE / RUL / Qty / Unit / Unit Cost /
#      Cycle Replace / Replace Percent / Year 1..N / Total Cost.
#      EBI, Bureau Veritas, AEI, Nova, NV5, Lender Consulting, EPIC (EPIC
#      drops the Cycle Replace and Replace Percent columns).
#   B  qty-in-eval     Partner Engineering. No replace-percent column; uses
#      "On Site Qty" + "Qty in Eval Period" instead.
#   C  base-cost       Nova variant keyed on "Base Cost".
#   D  numeric rating  EMG. Conditions as a 1-5 rating rather than words.
#   E  capital-considerations   Gabion. NO EUL/EFF AGE/RUL AT ALL. Columns are
#      Class / Item I.D. / Units / Number of Units / Unit Cost / Present Worth
#      / Start Year of Occurrence / Work Span Cycle / then CALENDAR-YEAR
#      columns carrying INFLATED dollars. Immediate vs reserve is the row's
#      class prefix: "I.N." (immediate need) vs "R.R." (replacement reserve).
#   F  R-numbered      Terracon. Item Description / EUL / Quantity / Units /
#      Cost / R-Total$ / Year 1..10 / Cumulative. No EFF AGE, no RUL, and EUL
#      quoted as a range.
#   G  inline cost summary   EBI, Atwell, Metropolitan Solutions. No single
#      consolidated table: each narrative section ends in its own small
#      "COST SUMMARY: Recommendation / EUL / EFF AGE / RUL / Year / Cost"
#      block, where Year is the term year the spend lands in (or "Immed").
#   H  narrative-costed  LandScience, CBC. No component table at all. Costs
#      appear as numbered items inside a section-level table with "Immed.
#      Cost" and "Reserve Cost" columns, or as a numbered "Immediate Need
#      Repairs Estimate" list. Component rows must be built from those items.
#   I  expected/remaining life   AEI (Homewood Suites HP set), Tetra Tech.
#      Same idea as A but the columns are spelled "Overall Expected Life" /
#      "Remaining Life" (AEI, both as RANGES) or "EXPECTED LIFE" /
#      "REFLECTIVE AGE" / "REMAINING LIFE" (Tetra Tech), and the year columns
#      are calendar years.

# ── CROSS-TABLE RECONCILIATION ─────────────────────────────────────────────
# These are the checks that make tabular extraction trustworthy: the stated
# totals must equal the sums of the pieces, across all three layers.
#   property.immediate_repairs_total_usd == Σ systems.immediate_repairs_usd
#                                        == Σ components[immediate].total_cost_usd
#   property.short_term_repairs_total_usd   == Σ components[short_term]
#   property.non_critical_repairs_total_usd == Σ components[non_critical]
#   property.reserves_total_uninflated_usd == Σ systems.replacement_reserves_usd
#                                          == Σ components[reserve].total_cost_usd
#   component: rul_years <= eul_years (when both numeric)
#   component[reserve]: Σ year_1..year_N ≈ total_cost_usd, unless
#                       years_inflated (Gabion's schedule is inflated,
#                       so it cannot equal an uninflated total)
RECONCILE_REL_TOL = 0.015   # 1.5% to absorb rounding across big tables

# Some firms add a CONTINGENCY on top of the line items, so a correct
# extraction can never sum to the stated total. Gabion prints the arithmetic
# in its own table footer:
#     Subtotals           $43,000
#     Contingency: 10.0%   $4,300
#     Escalated Totals:   $47,300
# The five immediate line items really do sum to $43,000 and the stated total
# really is $47,300; the $4,300 gap is the markup, not a missing row. Chasing
# it as an extraction bug is chasing a number that was never in the table.
# reconciliation_checks therefore accepts stated == summed + contingency as
# well as stated == summed, and only when the report states a contingency.

# Confidence at/below this routes a property field to the LLM judge.
CONFIDENCE_FLOOR = 0.75

# API limits for PDF requests: 100 pages / 32MB per request. extract.py keeps a
# front block and then scans the text layer for cost-table markers, so this is
# a page BUDGET rather than a blind "first N" slice. Reports whose tables sit
# behind the budget are still sent - it is the photo appendices that get cut.
#
# Raised from 50. At 50 this was a blind first-50 slice, and Wickliffe proved
# the cost was invisible: pages 169-172 of that 177-page Bureau Veritas report
# carry table markers and were never sent, which produces missing rows rather
# than an error. Now that pages are selected rather than truncated, headroom is
# nearly free - a report with nothing back there simply doesn't use it, and the
# typical send is the front block plus a handful of table pages. 90 keeps clear
# of the API's hard 100-page cap; extract.py enforces both that and the size
# limit, and says so on stderr when a report needs more room than it has.
MAX_PDF_PAGES = 90