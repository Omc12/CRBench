#!/usr/bin/env python3
"""Measure what the ASPLOS format checker measures: per-page text extent and
the font sizes actually used.

The CFP requires 1in top and bottom margins on US Letter, i.e. a text height
of at most 9in (648pt), and body text of at least 10pt. This reports the real
numbers from the compiled PDF instead of the values LaTeX was asked for, and
names the element responsible when a page is over.

    python paper/asplos/analysis/check_format.py [pdf]
"""
from __future__ import annotations

import collections
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
ASPLOS = os.path.dirname(HERE)

PT_PER_IN = 72.0
LIMIT_IN = 9.0                     # 11in page - 1in top - 1in bottom
TOP_MARGIN = 72.0                  # 1in
BOTTOM_EDGE = 792.0 - 72.0         # 1in from the bottom of an 11in page


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ASPLOS, "asplos.pdf")
    doc = fitz.open(path)
    print(f"{os.path.basename(path)}: {doc.page_count} pages, "
          f"page box {doc[0].rect.width:.0f} x {doc[0].rect.height:.0f} pt\n")

    print("page   top     bottom   height    over?   topmost / bottommost element")
    print("-" * 100)

    sizes = collections.defaultdict(set)          # rounded size -> pages
    examples = {}                                 # rounded size -> sample text

    for pno in range(doc.page_count):
        page = doc[pno]
        spans = []
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block.get("type") != 0:            # 0 = text
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    spans.append(span)
                    key = round(span["size"], 1)
                    sizes[key].add(pno + 1)
                    examples.setdefault(key, span["text"].strip()[:48])

        # Non-text ink (figures, rules) counts toward the visible text block.
        ink = [fitz.Rect(b["bbox"]) for b in d["blocks"] if b.get("type") == 1]
        for dr in page.get_drawings():
            ink.append(fitz.Rect(dr["rect"]))

        all_boxes = [fitz.Rect(s["bbox"]) for s in spans] + ink
        if not all_boxes:
            print(f"{pno+1:>4}   (empty page)")
            continue

        # Separate acmart's running head and folio from the body block. Both
        # legitimately live in the margin area; a checker that counts them
        # will over-measure every page of every acmart paper equally, so the
        # body block is what tells us whether we have a real problem.
        def is_furniture(rect):
            return rect.y1 < 60.0 or rect.y0 > 722.0

        boxes = [b for b in all_boxes if not is_furniture(b)] or all_boxes

        top = min(b.y0 for b in boxes)
        bot = max(b.y1 for b in boxes)
        height_in = (bot - top) / PT_PER_IN

        topmost = min(boxes, key=lambda b: b.y0)
        botmost = max(boxes, key=lambda b: b.y1)

        def describe(rect):
            for s in spans:
                if fitz.Rect(s["bbox"]) == rect:
                    return f'"{s["text"].strip()[:34]}" @{s["size"]:.1f}pt'
            return f"rule/graphic w={rect.width:.0f}pt h={rect.height:.0f}pt"

        flag = "OVER" if height_in > LIMIT_IN + 0.005 else "    "
        print(f"{pno+1:>4}  {top:6.1f}  {bot:6.1f}   {height_in:5.2f}in  {flag}   "
              f"{describe(topmost)}  ||  {describe(botmost)}")

    print()
    print("font sizes present (rounded to 0.1pt):")
    for size in sorted(sizes, reverse=True):
        pages = sorted(sizes[size])
        shown = pages[:12]
        more = "" if len(pages) <= 12 else f" +{len(pages)-12} more"
        print(f"  {size:5.1f}pt  pages {shown}{more}")
        print(f"           e.g. {examples[size]!r}")


if __name__ == "__main__":
    main()
