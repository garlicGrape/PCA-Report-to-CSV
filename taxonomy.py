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


# ══════════════════════════════════════════════════════════════════════════
# THE 12 SUBCATEGORIES - the canonical feature axis
# ══════════════════════════════════════════════════════════════════════════
# WHY THIS EXISTS
# `systems.csv` carried one row per numbered section of whatever the assessing
# firm happened to call things: 3,771 rows across 134 reports with **1,216
# distinct system_name strings**. "Exterior Walls" (55), "Building Frame" (42)
# and "Foundation/Substructure" (33) are the popular ones; the tail is a
# thousand one-off phrasings. That is not a feature - it is free text with a
# cost column attached, and it cannot be pooled across firms, which is the
# whole premise of leave-one-firm-out CV.
#
# These twelve are the ASTM E 2018 subcategories the owner specified. Every
# report now emits EXACTLY these twelve rows, present or absent, so the systems
# layer is a fixed-width 134 x 12 feature matrix instead of a ragged pile.
# Absent is recorded as a null condition on a row that still exists - "this
# report does not address vertical transportation" is a fact worth having, and
# a missing row cannot express it.
#
# The same twelve tag `components.csv`, so a line item and a condition rating
# can finally be joined on the same axis.

SUBCATEGORIES = [
    "site_improvements",
    "structural_frame_foundation",
    "building_envelope",
    "roofing",
    "mechanical_hvac",
    "plumbing",
    "electrical",
    "vertical_transportation",
    "fire_life_safety",
    "interior_elements",
    "accessibility",
    "additional_considerations",
]

# The sentinel. NOT one of the twelve, and deliberately not forced into them.
#
# "Utilities" / "Utility Providers and Special Systems" is 57 rows of the old
# corpus and describes which company supplies water, sewer, gas and power to
# the site. It spans plumbing and electrical and belongs cleanly to neither.
# Forcing it into `electrical` would corrupt the electrical feature on 57 rows
# to avoid an honest gap. Unmatched text lands here and is reported by
# `subcategory_coverage`, never guessed at.
SUBCATEGORY_OTHER = "other"

# What each subcategory covers, verbatim from the specification. This is the
# SINGLE SOURCE OF TRUTH: `extract.py` builds the prompt's definition list from
# this dict, so the schema and the instructions physically cannot drift apart -
# which is how the old systems layer ended up with 1,216 names.
SUBCATEGORY_SCOPE = {
    "site_improvements":
        "paving, curbs, sidewalks, parking, drainage, retaining walls, "
        "landscaping, site lighting, fencing, and signage",
    "structural_frame_foundation":
        "footings, slabs, columns, beams, load bearing walls, and floor and "
        "roof framing",
    "building_envelope":
        "exterior walls, cladding, windows, doors, sealants, and waterproofing",
    "roofing":
        "membrane or shingles, flashing, drainage, gutters and downspouts, "
        "and remaining useful life",
    "mechanical_hvac":
        "rooftop units, split systems, boilers, chillers, ductwork, and "
        "controls",
    "plumbing":
        "domestic water supply and piping, water heaters, sanitary and storm "
        "piping, and fixtures",
    "electrical":
        "service and distribution, panels, wiring type, lighting, and "
        "emergency generator",
    "vertical_transportation":
        "elevators, lifts, and escalators, including modernization status",
    "fire_life_safety":
        "sprinklers, fire alarm, extinguishers, emergency lighting, egress, "
        "and standpipes",
    "interior_elements":
        "common area and unit finishes, flooring, walls, ceilings, cabinetry, "
        "appliances, and amenity spaces",
    "accessibility":
        "a limited compliance screen (ADA and FHA) of accessible routes, "
        "parking, entrances, restrooms, and units",
    "additional_considerations":
        "commercial kitchen equipment, pools, laundry, and, for senior housing "
        "specifically, nurse call systems, generators, and commercial kitchen "
        "and dining infrastructure",
}

assert len(SUBCATEGORIES) == 12
assert set(SUBCATEGORIES) == set(SUBCATEGORY_SCOPE)

# ── free text -> subcategory ───────────────────────────────────────────────
# ORDER IS THE DISAMBIGUATION MECHANISM, exactly as in _RULES above. Several
# of these orderings are load-bearing and were chosen against the real corpus
# vocabulary; changing the order silently re-buckets rows:
#
#   site_improvements BEFORE electrical    - "Site Lighting" (28 rows) and
#       "Exterior Lights" (19) are site work in the spec, not electrical.
#   roofing BEFORE site_improvements       - "Roof Drainage" (37) is roofing;
#       "Storm Water Drainage" (37) is site. Both are "drainage".
#   site_improvements BEFORE structural    - "Retaining Walls" (18) and
#       "Perimeter Walls, Gates, and Fences" (16) are site in the spec.
#   additional_considerations BEFORE interior - the spec puts "amenity spaces"
#       under interior but pools/laundry/commercial kitchen under additional.
#       Generic "Amenities" (17+15 rows) resolves to additional.
#   mechanical_hvac BEFORE plumbing        - "Cooling Tower" is HVAC.
#   structural BEFORE building_envelope    - "Building Stairs and Balconies"
#       (20) is frame, and envelope's bare `wall` would otherwise claim it.
#
# Stems carry a trailing \w* for the reason documented on _RULES: a closing \b
# after a stem defeats the stem.
_SUBCAT_RULES = [
    ("accessibility",
     r"\b(?:ada\b|a\.d\.a|americans with disabilit|fair housing|fha\b|"
     r"accessib|barrier[- ]free|path of travel|ansi\b)\w*"),

    ("vertical_transportation",
     r"\b(?:elevator|escalator|wheelchair lift|chair ?lift|vertical "
     r"transport|transportation system|hoist|dumbwaiter|modernizat)\w*"),

    ("fire_life_safety",
     r"\b(?:fire(?!place)|sprinkler|standpipe|extinguish|smoke detect|"
     r"alarm|life safety|egress|emergency light|suppression|"
     r"exit sign|hydrant)\w*"),

    ("additional_considerations",
     r"\b(?:pool|spa(?![a-z])|jacuzzi|sauna|laundry|fitness|clubhouse|"
     r"playground|tennis|basketball|commercial kitchen|kitchen equip|"
     r"dining (?:room|facilit|infrastructur)|nurse ?call|"
     r"amenit|special feature|grill|salon|barber|theater|"
     r"dog park|car ?wash|recreation)\w*"),

    ("roofing",
     r"\b(?:roof(?!top)|shingle|tpo\b|epdm|pvc roof|built.?up|membrane|"
     r"parapet|gutter|downspout|flashing|skylight|soffit|fascia|"
     r"pitch pocket|coping)\w*"),

    ("site_improvements",
     r"\b(?:pav(?:e|ing|ement)|asphalt|concrete flatwork|flatwork|curb|"
     r"sidewalk|walkway|parking|driveway|drive lane|seal ?coat|flat.?work|"
     r"crack ?seal|(?:re)?strip(?:e|ing)|site light|exterior light|"
     r"site improvement|site work|sitework|landscap|irrigation|"
     r"topograph|grading|retaining wall|perimeter wall|fenc|gate\b|"
     r"signage|monument sign|storm ?water|drainage|erosion|"
     r"catch basin|auxiliary structure|carport|patio|plaza|"
     r"mail ?(?:box|kiosk)|dumpster|waste (?:storage|enclosure)|trash|"
     r"refuse|ground(?:s|water)|appurtenance|site(?![a-z]))\w*"),

    ("mechanical_hvac",
     r"\b(?:hvac|heating|ventilat|air ?condition|cooling|boiler|chiller|"
     r"furnace|rtu\b|rooftop unit|split.?system|heat pump|ptac|"
     r"packaged terminal|air handl|ahu\b|fan coil|duct|thermostat|"
     r"exhaust fan|cooling tower|make.?up air|condens|mechanical|"
     r"building (?:management|automation)|bms\b|ems\b)\w*"),

    ("plumbing",
     r"\b(?:plumb|domestic (?:water|hot water)|water heater|water supply|"
     r"sewer|sanitary|sump|riser|backflow|grease trap|lavatory|toilet|"
     r"water main|water line|natural gas|gas (?:line|piping|service)|"
     r"pip(?:e|ing)|fixture)\w*"),

    ("electrical",
     r"\b(?:electric|panel ?board|panel\b|switch ?gear|transformer|"
     r"wiring|conduit|receptacle|distribution|service entrance|"
     r"generator|lighting|light fixture|meter\b|bus ?(?:way|duct))\w*"),

    ("structural_frame_foundation",
     r"\b(?:foundation|substructure|superstructure|structur|"
     r"frame|framing|joist|column|beam|footing|slab|"
     r"load.?bearing|settlement|stair|balcon|deck\b|"
     r"crawl ?space|basement|wood destroying|termite)\w*"),

    ("building_envelope",
     r"\b(?:envelope|exterior wall|wall|cladding|siding|facade|façade|"
     r"stucco|eifs|masonry|brick|veneer|window|door|glazing|curtain ?wall|"
     r"sealant|caulk|waterproof|weatherproof|moisture|water intrusion|"
     r"microbial|mold|mildew|repoint|tuckpoint|canopy|awning|"
     r"exterior (?:paint|finish))\w*"),

    ("interior_elements",
     r"\b(?:interior|finish|common area|unit finish|guest ?room|"
     r"cabinet|counter|vanity|carpet|floor|tile|drywall|gypsum|"
     r"ceiling|corridor|lobby|ff&e|ffe\b|furnish|appliance|"
     r"refrigerat|range\b|oven|dishwash|microwave|washer|dryer|"
     r"millwork|blind|window covering|renovat|refurbish|"
     r"kitchen|bath(?:room)?|soft goods|"
     r"(?:tenant|support|amenity) (?:space|area))\w*"),
]

_SUBCAT_COMPILED = [(c, re.compile(p, re.I)) for c, p in _SUBCAT_RULES]

# The sixteen component categories fold into the twelve. `professional_services`
# (surveys, studies, permits, management fees) has no physical subcategory - it
# is a cost line, not a building system - so it goes to the sentinel rather
# than being smuggled into one of the twelve.
COMPONENT_CATEGORY_TO_SUBCATEGORY = {
    "roofing":               "roofing",
    "paving_site":           "site_improvements",
    "landscaping":           "site_improvements",
    "hvac":                  "mechanical_hvac",
    "plumbing":              "plumbing",
    "electrical":            "electrical",
    "elevator":              "vertical_transportation",
    "envelope":              "building_envelope",
    "structural":            "structural_frame_foundation",
    "fire_life_safety":      "fire_life_safety",
    "interior_finishes":     "interior_elements",
    "appliances":            "interior_elements",
    "accessibility":         "accessibility",
    "amenity":               "additional_considerations",
    "professional_services": SUBCATEGORY_OTHER,
    "other":                 SUBCATEGORY_OTHER,
}

assert set(COMPONENT_CATEGORY_TO_SUBCATEGORY) == set(CATEGORIES)
assert set(COMPONENT_CATEGORY_TO_SUBCATEGORY.values()) <= (
    set(SUBCATEGORIES) | {SUBCATEGORY_OTHER})


def classify_subcategory(text) -> str:
    """Free text -> one of SUBCATEGORIES, or SUBCATEGORY_OTHER.

    Used for firm-specific system names ("Foundation/Substructure",
    "Walkways, Grade-Level Steps and Ramps") and for any other loose phrase
    that needs to land on the canonical axis.
    """
    if not isinstance(text, str) or not text.strip():
        return SUBCATEGORY_OTHER
    for sub, rx in _SUBCAT_COMPILED:
        if rx.search(text):
            return sub
    return SUBCATEGORY_OTHER


def explain_subcategory(text) -> tuple:
    """-> (subcategory, matched text). For auditing the mapping."""
    if not isinstance(text, str) or not text.strip():
        return (SUBCATEGORY_OTHER, None)
    for sub, rx in _SUBCAT_COMPILED:
        m = rx.search(text)
        if m:
            return (sub, m.group(0))
    return (SUBCATEGORY_OTHER, None)


def subcategory_for_component(description) -> str:
    """Component description -> one of the twelve, via the 16-category rules.

    Routed through `classify` on purpose rather than running the system-name
    patterns directly: the component rules are tuned to line-item phrasing
    ("EPDM roof replacement", "Replace TPO roofing system") and have been
    exercised against 2,409 real rows. Re-deriving that work would be a second
    place for the mapping to be wrong.
    """
    return COMPONENT_CATEGORY_TO_SUBCATEGORY[classify(description)]


def subcategory_coverage(texts, classifier=classify_subcategory) -> dict:
    """How much of a column the twelve claim, and what they miss.

    Same contract as `coverage_report`. Run it after every corpus change: a
    rising `other` share means new vocabulary has arrived and the tail needs a
    look before the systems layer is trusted as a feature.
    """
    subs = [classifier(t) for t in texts]
    counts = Counter(subs)
    n = max(1, len(subs))
    unmatched = [t for t, s in zip(texts, subs) if s == SUBCATEGORY_OTHER]
    return {"n": len(subs), "counts": counts,
            "matched_pct": round(100 * (n - counts[SUBCATEGORY_OTHER]) / n, 1),
            "unmatched_examples": unmatched[:25]}
