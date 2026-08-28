"""
20-second check that LangSmith is wired up.

Run:  python smoke.py
Then open smith.langchain.com and look in the `pca-extract` project — you
should see a `hello` trace. If it's there, every @traceable in the real
pipeline will log the same way and you're ready to go.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import traceable


@traceable
def hello(x: str) -> str:
    return x.upper()


if __name__ == "__main__":
    key = os.getenv("LANGSMITH_API_KEY")
    proj = os.getenv("LANGSMITH_PROJECT")
    tracing = os.getenv("LANGSMITH_TRACING")

    if not key:
        print("✗ LANGSMITH_API_KEY not found. Did you fill in .env?")
    elif tracing != "true":
        print("✗ LANGSMITH_TRACING is not 'true' in .env — traces won't be sent.")
    else:
        result = hello("langsmith is wired")
        print(f"✓ ran hello() -> {result}")
        print(f"✓ tracing to project: {proj}")
        print("Now refresh smith.langchain.com and open that project to see the trace.")
