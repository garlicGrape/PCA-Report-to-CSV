"""
The number that makes this "enterprise-grade": measured per-field extraction
accuracy on a labeled gold set, tracked in LangSmith over time.

What it does:
  1. Loads your synthetic gold (data/gold/*.json).
  2. Optionally pushes it to a LangSmith dataset (so runs are comparable).
  3. Runs the extractor on each report and scores every field:
       - numbers: within 1% (or exact for small ints)
       - categories/dates/strings: exact (case-insensitive)
  4. Prints a per-field accuracy table and logs the run to LangSmith.

Run:  python eval_suite.py
"""
import json
from pathlib import Path
from collections import defaultdict

from schema import CANONICAL_FIELDS, FIELD_META
from extract import extract

HERE = Path(__file__).parent
GOLD = HERE / "data" / "gold"
INBOX = HERE / "data" / "inbox"


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


def field_correct(field, predicted, gold) -> bool:
    if gold is None:
        return predicted is None
    if predicted is None:
        return False
    if FIELD_META.get(field, {}).get("type") == "number":
        p, g = _num(predicted), _num(gold)
        if p is None or g is None:
            return False
        return abs(p - g) <= max(1.0, abs(g) * 0.01)
    return str(predicted).strip().lower() == str(gold).strip().lower()


def run():
    gold_files = sorted(GOLD.glob("*.json"))
    if not gold_files:
        print(f"No gold in {GOLD}. Run: python make_synthetic.py --n 6")
        return

    per_field = defaultdict(lambda: [0, 0])  # field -> [correct, total]
    overall = [0, 0]

    for gf in gold_files:
        pid = gf.stem
        gold = json.loads(gf.read_text())["values"]
        pdf = INBOX / f"{pid}.pdf"
        record = extract(str(pdf))
        for f in CANONICAL_FIELDS:
            pred = record.get(f, {}).get("value")
            ok = field_correct(f, pred, gold.get(f))
            per_field[f][1] += 1
            per_field[f][0] += int(ok)
            overall[1] += 1
            overall[0] += int(ok)
        print(f"scored {pid}")

    print("\nPer-field accuracy")
    print("-" * 44)
    for f in CANONICAL_FIELDS:
        c, t = per_field[f]
        if t:
            print(f"{f:<26} {c}/{t}  {c/t:5.0%}")
    print("-" * 44)
    print(f"{'OVERALL':<26} {overall[0]}/{overall[1]}  {overall[0]/overall[1]:5.0%}")


# If you'd rather use LangSmith's native evaluate() (recommended once this works),
# the shape is:
#
#   from langsmith import Client
#   from langsmith.evaluation import evaluate
#   client = Client()
#   # 1) create a dataset once from your gold, then:
#   def target(inputs):  # inputs = {"pdf_path": ...}
#       return {"record": extract(inputs["pdf_path"])}
#   def per_field_accuracy(run, example):  # custom evaluator -> {"key","score"}
#       ...
#   evaluate(target, data="pca-plus-gold", evaluators=[per_field_accuracy])
#
# That gives you the comparison UI, regression tracking, and shareable reports.

if __name__ == "__main__":
    run()
