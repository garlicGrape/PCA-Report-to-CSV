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
"""

CONDITIONS = ["excellent", "good", "fair", "poor", "na"]

# ── PROPERTY (one row per report) ──────────────────────────────────────────
PROPERTY_FIELDS = [
    "property_name", "property_address", "city", "state", "zip",
    "report_firm", "project_number", "client_name",
    "report_date", "assessment_date",
    "property_type", "year_built", "renovation_years",     # e.g. "1985; 2018"
    "num_stories", "num_units", "num_residential_units", "num_commercial_units",
    "net_rentable_sf", "site_acres", "num_parcels", "num_buildings",
    "basement",                                            # e.g. "full, unfinished"
    "construction_type", "facade_materials", "roof_types",
    "heating_source", "cooling_type", "water_heater_type",
    "num_elevators", "fire_sprinklers", "emergency_generator",
    "overall_condition", "overall_rul_years",
    "immediate_repairs_total_usd", "immediate_repairs_per_unit_usd",
    "reserves_total_uninflated_usd", "reserves_total_inflated_usd",
    "reserves_per_unit_usd", "inflation_rate_pct", "reserve_term_years",
    "reserve_per_sf_year_uninflated", "reserve_per_sf_year_inflated",
    "reserve_per_unit_year_uninflated", "reserve_per_unit_year_inflated",
]

PROPERTY_META = {
    "report_date":        {"type": "date"},
    "assessment_date":    {"type": "date"},
    "year_built":         {"type": "number", "min": 1850, "max": 2026},
    "num_stories":        {"type": "number", "min": 1, "max": 120},
    "num_units":          {"type": "number", "min": 1, "max": 5000},
    "num_residential_units": {"type": "number", "min": 0, "max": 5000},
    "num_commercial_units":  {"type": "number", "min": 0, "max": 1000},
    "net_rentable_sf":    {"type": "number", "min": 1000, "max": 10_000_000},
    "site_acres":         {"type": "number", "min": 0.01, "max": 2000},
    "num_parcels":        {"type": "number", "min": 1, "max": 100},
    "num_elevators":      {"type": "number", "min": 0, "max": 60},
    "overall_condition":  {"type": "category", "allowed": CONDITIONS},
    "overall_rul_years":  {"type": "number", "min": 0, "max": 100},
    "immediate_repairs_total_usd":    {"type": "number", "min": 0, "max": 150_000_000},
    "reserves_total_uninflated_usd":  {"type": "number", "min": 0, "max": 150_000_000},
    "reserves_total_inflated_usd":    {"type": "number", "min": 0, "max": 200_000_000},
    "inflation_rate_pct": {"type": "number", "min": 0, "max": 15},
    "reserve_term_years": {"type": "number", "min": 1, "max": 30},
}

# ── SYSTEMS (one row per numbered section) ─────────────────────────────────
# From the Executive Summary Table: 2.1 Topography .. 5.2 Fire Department.
SYSTEM_FIELDS = [
    "section_code",            # "3.4"
    "system_name",             # "Roofing"
    "condition",               # primary rating (lowercase)
    "condition_secondary",     # second X when rated e.g. "Good to Fair", else null
    "action_required",         # verbatim-ish: "Replace", "Refurbish, Repair", "None"
    "immediate_repairs_usd",   # from the exec summary row; null if blank
    "replacement_reserves_usd",
]

SYSTEM_META = {
    "condition":            {"type": "category", "allowed": CONDITIONS},
    "condition_secondary":  {"type": "category", "allowed": CONDITIONS},
    "immediate_repairs_usd":    {"type": "number", "min": 0, "max": 50_000_000},
    "replacement_reserves_usd": {"type": "number", "min": 0, "max": 50_000_000},
}

# ── COMPONENTS (one row per Table 1 / Table 2 line item) ───────────────────
COMPONENT_FIELDS = [
    "table",                # "immediate" (Table 1) or "reserve" (Table 2)
    "section_code",         # "3.4"
    "description",          # "EPDM roof replacement"
    "eul_years",            # Table 2 only; null on Table 1 rows
    "effective_age_years",  # null when report says "var"/"Varies"
    "rul_years",            # null when "var"/"Varies"
    "rul_varies",           # true when the report said var/varies
    "quantity", "unit",     # 18800, "SF" | "ALW" | "EA" | "LF" | "UNIT"
    "unit_cost_usd",
    "cycle_replace_cost_usd",
    "replace_percent",      # 100, 200, 1200 ... (recurring items exceed 100)
    "total_cost_usd",
    # 12-year spend schedule from Table 2 (null on Table 1 rows / empty years)
    "year_1", "year_2", "year_3", "year_4", "year_5", "year_6",
    "year_7", "year_8", "year_9", "year_10", "year_11", "year_12",
]

COMPONENT_META = {
    "table":               {"type": "category", "allowed": ["immediate", "reserve"]},
    "eul_years":           {"type": "number", "min": 0, "max": 100},
    "effective_age_years": {"type": "number", "min": 0, "max": 100},
    "rul_years":           {"type": "number", "min": 0, "max": 100},
    "quantity":            {"type": "number", "min": 0, "max": 10_000_000},
    "unit_cost_usd":       {"type": "number", "min": 0, "max": 10_000_000},
    "cycle_replace_cost_usd": {"type": "number", "min": 0, "max": 50_000_000},
    "replace_percent":     {"type": "number", "min": 0, "max": 5000},
    "total_cost_usd":      {"type": "number", "min": 0, "max": 50_000_000},
}

# ── CROSS-TABLE RECONCILIATION ─────────────────────────────────────────────
# These are the checks that make tabular extraction trustworthy: the stated
# totals must equal the sums of the pieces, across all three layers.
#   property.immediate_repairs_total_usd == Σ systems.immediate_repairs_usd
#                                        == Σ components[immediate].total_cost_usd
#   property.reserves_total_uninflated_usd == Σ systems.replacement_reserves_usd
#                                          == Σ components[reserve].total_cost_usd
#   component: rul_years <= eul_years (when both numeric)
#   component[reserve]: Σ year_1..year_12 ≈ total_cost_usd
RECONCILE_REL_TOL = 0.015   # 1.5% to absorb rounding across big tables

# Confidence at/below this routes a property field to the LLM judge.
CONFIDENCE_FLOOR = 0.75

# API limits for PDF requests: 100 pages / 32MB per request. The narrative +
# tables in these reports end well before the photo appendices, so we slice.
MAX_PDF_PAGES = 50
