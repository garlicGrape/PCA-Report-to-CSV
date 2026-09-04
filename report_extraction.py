"""Roster of every report in the dataset, by source PDF filename.

Reads data/aggregate/manifest.csv (one row per report the run produced) and
joins the property identity from properties.csv. Run after batch.py.
"""
import sys
from pathlib import Path
import pandas as pd

AGG = Path(__file__).parent / "data" / "aggregate"


def main():
    man = pd.read_csv(AGG / "manifest.csv")
    props = pd.read_csv(AGG / "properties.csv")
    keep = ["property_id", "property_name", "city", "state", "report_firm",
            "prompt_version"]
    df = man.merge(props[[c for c in keep if c in props.columns]],
                   on="property_id", how="left")

    df["pdf"] = df["source_file"].fillna(df["property_id"])
    df = df.sort_values("pdf", key=lambda s: s.str.lower())

    print(f"{'#':>4}  {'PDF FILE':<62} {'STATUS':<13} {'FIRM':<22} PROPERTY")
    print("-" * 150)
    for i, r in enumerate(df.itertuples(), 1):
        name = str(getattr(r, "property_name", "") or "")
        loc = ", ".join(str(x) for x in
                        (getattr(r, "city", ""), getattr(r, "state", ""))
                        if x and str(x) != "nan")
        who = f"{name} ({loc})" if loc else name
        print(f"{i:>4}  {str(r.pdf)[:62]:<62} {str(r.status):<13} "
              f"{str(getattr(r, 'report_firm', ''))[:22]:<22} {who[:44]}")

    print("-" * 150)
    print(f"TOTAL: {len(df)} reports   "
          f"{dict(df['status'].value_counts())}")
    if "prompt_version" in df.columns:
        print(f"prompt versions: {dict(df['prompt_version'].fillna('unversioned').value_counts())}")

    out = AGG / "extracted_reports.csv"
    df[["pdf"] + [c for c in keep if c in df.columns] +
       ["status", "seconds"]].to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
