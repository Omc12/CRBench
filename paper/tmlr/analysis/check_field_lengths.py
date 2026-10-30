#!/usr/bin/env python3
"""Check the OpenReview form fields against their 5000-character limits.

OpenReview counts characters, not words, and silently reports a negative
remaining count once you are over. Run this before pasting.

Usage:  python paper/tmlr/analysis/check_field_lengths.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "OPENREVIEW_FIELDS.md")

LIMIT = 5000
FIELDS = {
    "Abstract": LIMIT,
    "Changes Since Last Submission": LIMIT,
    "Title": 250,
}


def blocks(text):
    """Map each '## Heading' to the first fenced block beneath it."""
    out = {}
    for m in re.finditer(r"^## (.+?)$(.*?)(?=^## |\Z)", text, re.M | re.S):
        heading, body = m.group(1).strip(), m.group(2)
        fence = re.search(r"```\n(.*?)```", body, re.S)
        if fence:
            out[heading] = fence.group(1).rstrip("\n")
    return out


def main():
    text = open(DOC, encoding="utf-8").read()
    found = blocks(text)
    bad = False
    for name, limit in FIELDS.items():
        if name not in found:
            print(f"  MISSING  {name}")
            bad = True
            continue
        n = len(found[name])
        ok = n <= limit
        bad = bad or not ok
        print(f"  {'ok  ' if ok else 'OVER'}  {name:<32} {n:>5} / {limit} chars"
              f"{'' if ok else f'  ({n - limit} over)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
