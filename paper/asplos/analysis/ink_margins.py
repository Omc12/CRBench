#!/usr/bin/env python3
"""Measure the margins from the rendered pixels, not the PDF object model.

Vector paths inside embedded figures carry clip rectangles with absurd
coordinates, so measuring the PDF's drawing objects gives nonsense. Rendering
each page and finding the non-white pixels measures what a reader — and a
format checker looking at the page — actually sees.

Reports the full ink box, and the ink box excluding acmart's running head and
folio, which live in the margin area by the template's design.

    python paper/asplos/analysis/ink_margins.py [pdf]
"""
from __future__ import annotations

import os
import sys

import fitz

DPI = 150
SCALE = DPI / 72.0
NEED_SIDE = 0.75          # inches
NEED_VERT = 1.00          # inches

# Bands that hold acmart furniture: the running head and the folio/line
# numbers. Measured from the compiled file: header 45-58pt, folio ~720-730pt.
HEAD_BAND = 62.0
FOOT_BAND = 719.0


def ink_box(pix, x_from=0, y_from=0, y_to=None):
    """Bounding box of non-white pixels, in PDF points."""
    w, h, n = pix.width, pix.height, pix.n
    data = pix.samples
    y_to = h if y_to is None else min(h, y_to)
    x0, y0, x1, y1 = w, h, -1, -1
    for y in range(y_from, y_to):
        row = data[y * pix.stride:(y + 1) * pix.stride]
        for x in range(x_from, w):
            p = row[x * n:x * n + 3]
            if p[0] < 245 or p[1] < 245 or p[2] < 245:
                if x < x0: x0 = x
                if x > x1: x1 = x
                if y < y0: y0 = y
                if y > y1: y1 = y
    if x1 < 0:
        return None
    return (x0 / SCALE, y0 / SCALE, (x1 + 1) / SCALE, (y1 + 1) / SCALE)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asplos.pdf")
    doc = fitz.open(path)
    W, H = doc[0].rect.width, doc[0].rect.height
    print(f"{os.path.basename(path)}  ({W:.0f} x {H:.0f} pt), rendered at {DPI} dpi")
    print(f"required: side >= {NEED_SIDE}in, top/bottom >= {NEED_VERT}in, "
          f"text height <= 9.00in\n")
    print("page |            full ink box            |   excluding head/folio")
    print("     |  left   right    top   bottom  hgt |  top  bottom   height")
    print("-" * 78)

    worst = {}
    for pno in range(doc.page_count):
        page = doc[pno]
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))

        full = ink_box(pix)
        body = ink_box(pix, y_from=int(HEAD_BAND * SCALE),
                       y_to=int(FOOT_BAND * SCALE))
        if full is None:
            continue

        fl, ft, fr, fb = full[0], full[1], W - full[2], H - full[3]
        fh = (full[3] - full[1]) / 72.0
        if body:
            bt, bb = body[1], H - body[3]
            bh = (body[3] - body[1]) / 72.0
        else:
            bt = bb = bh = float("nan")

        print(f"{pno+1:>4} | {fl:6.1f} {fr:6.1f} {full[1]:6.1f} {fb:6.1f} "
              f"{fh:5.2f}in | {bt:5.1f} {bb:6.1f}  {bh:5.2f}in")

        for k, v in (("left", fl), ("right", fr),
                     ("body top", bt), ("body bottom", bb)):
            if k not in worst or v < worst[k][0]:
                worst[k] = (v, pno + 1)
        k = "body height"
        if k not in worst or bh > worst[k][0]:
            worst[k] = (bh, pno + 1)

    print()
    for k in ("left", "right", "body top", "body bottom"):
        v, p = worst[k]
        need = NEED_SIDE if k in ("left", "right") else NEED_VERT
        ok = "OK  " if v / 72.0 >= need - 0.01 else "FAIL"
        print(f"  {ok} smallest {k:<12} = {v:6.1f}pt = {v/72:.3f}in "
              f"(need {need:.2f}in)  worst page {p}")
    v, p = worst["body height"]
    ok = "OK  " if v <= 9.005 else "FAIL"
    print(f"  {ok} largest  body height  = {v:.2f}in "
          f"(limit 9.00in)  worst page {p}")


if __name__ == "__main__":
    main()
