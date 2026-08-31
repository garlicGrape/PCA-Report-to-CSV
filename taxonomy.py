"""
Component description -> canonical taxonomy.

WHY THIS EXISTS
The timing layer trains on components.csv, and pooling across firms is what
makes it work: you cannot learn an EUL for "roof membrane" from one report.
But `description` is free text written by whoever assessed the property, and
on the first 258 extracted rows there were 251 DISTINCT strings - only 7
appeared more than once. "EPDM roof replacement" (EBI), "Replace TPO roofing
system" (AEI) and "Roof Replacement - Shingled Roofs (1985 vintage)" (Gabion)
are the same component to a model and three unrelated strings to a computer.

DESIGN
Deterministic and auditable on purpose. Every row's category can be traced to
the rule that assigned it, which is what makes the mapping defensible and
reproducible across re-runs; an LLM pass is reserved for the residual tail and
must be cached, never re-decided per run.

`section_code` is deliberately NOT used as a signal. It is 100% populated but
firm-specific and non-comparable: Bureau Veritas writes "5.2", Gabion
"B.4.1A", Terracon "R-1", CBC "Site", and on reports whose tables carry no
section numbers the extractor emits its own slugs ("life_safety", "mep").
Mapping it needs a per-firm lookup - worth building later, useless as-is.

USAGE
    from taxonomy import classify, coverage_report
    df["component_category"] = df["description"].map(classify)
"""
import re
from collections import Counter

# Canonical categories. Deliberately coarse: these are the levels at which a
# useful life is actually shared. Splitting finer (asphalt-overlay vs
# asphalt-sealcoat) is a second pass once there are enough rows per bucket to
# support it - at ~3,100 corpus rows a 40-bucket taxonomy averages 75 rows.
CATEGORIES = [
    "roofing", "paving_site", "hvac", "plumbing", "electrical",
    "elevator", "envelope", "interior_finishes", "appliances",
    "fire_life_safety", "structural", "accessibility", "amenity",
    "landscaping", "professional_services", "other",
]

# Ordered most-specific first: the first rule that matches wins, so a phrase
# that could read two ways is resolved by whichever bucket is narrower.
# "pool heater" is an amenity, not HVAC; "fire pump" is life safety, not
# plumbing. Order is the disambiguation mechanism - keep it.
# NOTE ON THE PATTERNS: terms are STEMS and are matched with a trailing \\w*,
# because a closing \\b after a stem defeats the stem. "refrigerat\\b" does not
# match "Refrigerator"; "floor\\b" does not match "flooring"; "inspect\\b" does
# not match "inspection". That mistake silently pushed a fifth of the rows
# into "other" on the first version of this file. Where a suffix WOULD cause a
# false positive the term is anchored explicitly with (?![a-z]) instead.
_RULES = [
    ("professional_services", r"\b(?:survey|stud(?:y|ies)|report|inspect|"
                              r"evaluat|assess|engineer|consultant|test|"
                              r"analys|analyz|audit|review|obtain|document|"
                              r"design|permit|management fee)\w*"),
    ("accessibility",         r"\b(?:ada\b|accessib|barrier|path of travel|"
                              r"ansi\b)\w*"),
    ("fire_life_safety",      r"\b(?:fire(?!place)|sprinkler|alarm|extinguish|"
                              r"smoke|standpipe|life safety|egress|"
                              r"emergency light)\w*"),
    ("elevator",              r"\b(?:elevator|escalator|hoist)\w*"),
    ("appliances",            r"\b(?:refrigerat|range\b|oven|cooktop|dishwash|"
                              r"microwave|washer|dryer|disposal|freezer|"
                              r"appliance|laundry equip)\w*"),
    ("roofing",               r"\b(?:roof(?!top)|shingle|tpo\b|epdm|membrane|parapet|"
                              r"gutter|downspout|flashing|skylight|soffit|"
                              r"fascia|pitch pocket)\w*"),
    ("paving_site",           r"\b(?:asphalt|pavement|paving|seal ?coat|"
                              r"(?:re)?strip|curb|sidewalk|flatwork|"
                              r"parking|driveway|crack ?seal|walkway)\w*"),
    ("hvac",                  r"\b(?:hvac|condens|furnace|boiler|chiller|rtu|"
                              r"ptac|heat pump|air handl|fan coil|split.?system|"
                              r"rooftop|packaged terminal|duct|thermostat|"
                              r"ventilat|exhaust fan|make.?up air|"
                              r"cooling tower|air condition|ahu\b)\w*"),
    ("plumbing",              r"\b(?:water heater|plumb|domestic water|riser|"
                              r"sewer|sanitary|piping|water line|toilet|"
                              r"lavatory|backflow|water main|grease trap|water (?:storage|tank)|expansion tank)\w*"),
    ("electrical",            r"\b(?:electric|panelboard|switchgear|"
                              r"transformer|generator|lighting|light fixture|"
                              r"wiring|receptacle|distribution)\w*"),
    ("envelope",              r"\b(?:window|door|siding|facade|fa\u00e7ade|stucco|"
                              r"masonry|brick|caulk|sealant|balcon|cladding|"
                              r"weatherproof|repoint|tuckpoint|exterior wall|"
                              r"canopy|trim|paint)\w*"),
    ("structural",            r"\b(?:foundation|structural|framing|joist|"
                              r"column|slab|retaining wall|settlement)\w*"),
    ("amenity",               r"\b(?:pool|spa\b|fitness|clubhouse|playground|"
                              r"tennis|grill|patio furniture|mail|signage|"
                              r"shuttle|salon|dining room|kitchen equip)\w*"),
    ("landscaping",           r"\b(?:landscap|irrigation|tree|shrub|lawn|"
                              r"fenc|gate|planting|mulch)\w*"),
    ("interior_finishes",     r"\b(?:carpet|floor|finish|cabinet|counter|"
                              r"vanity|tile|drywall|ceiling|corridor|"
                              r"common area|renovat|refurbish|furnish|"
                              r"ff&e|blind|window covering|vinyl|guestroom|"
                              r"unit interior)\w*"),
]

_COMPILED = [(cat, re.compile(pat, re.I)) for cat, pat in _RULES]


def classify(description) -> str:
    """-> a CATEGORIES value. Unmatched text becomes "other", never a guess."""
    if not isinstance(description, str) or not description.strip():
        return "other"
    for cat, rx in _COMPILED:
        if rx.search(description):
            return cat
    return "other"


def explain(description) -> tuple:
    """-> (category, the pattern that matched). For auditing the mapping."""
    if not isinstance(description, str) or not description.strip():
        return ("other", None)
    for cat, rx in _COMPILED:
        m = rx.search(description)
        if m:
            return (cat, m.group(0))
    return ("other", None)


def coverage_report(descriptions) -> dict:
    """How much of a column the rules actually claim, and what they miss.

    Run this after every corpus change. A rising "other" share means a firm
    with unfamiliar phrasing has entered the set, and the tail needs another
    look before the taxonomy is trusted as a model feature.
    """
    cats = [classify(d) for d in descriptions]
    counts = Counter(cats)
    n = max(1, len(cats))
    unmatched = [d for d, c in zip(descriptions, cats) if c == "other"]
    return {"n": len(cats), "counts": counts,
            "matched_pct": round(100 * (n - counts["other"]) / n, 1),
            "unmatched_examples": unmatched[:25]}
