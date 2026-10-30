#!/usr/bin/env python3
"""Emit the plain-text title and abstract for the HotCRP submission form.

HotCRP fields are plain text, not LaTeX, so this strips the markup and maps
maths to the Unicode a form field can hold.

    python paper/asplos/analysis/make_hotcrp_fields.py
"""
from __future__ import annotations

import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ASPLOS = os.path.dirname(HERE)
TEX = os.path.join(ASPLOS, "asplos.tex")

SUBS = [
    (r"\\sys\{\}", "TriAxis"),
    (r"\\emph\{([^}]*)\}", r"\1"),
    (r"\\textbf\{([^}]*)\}", r"\1"),
    (r"\\times", "\u00d7"),
    (r"\\rho", "\u03c1"),
    (r"\\infty", "\u221e"),
    (r"\\%", "%"),
    (r"\\,", " "),
    (r"\{,\}", ","),
    (r"\{\+\}", "+"),
    (r"\{-\}", "\u2212"),
    (r"---", "\u2014"),
    (r"--", "\u2013"),
    (r"\$", ""),
    (r"~", " "),
    (r"\{", ""),
    (r"\}", ""),
]


def strip_tex(text):
    for pat, rep in SUBS:
        text = re.sub(pat, rep, text)
    return text


def main():
    src = io.open(TEX, encoding="utf-8").read()

    title = src.split("\\title{", 1)[1].split("\n\n", 1)[0]
    title = strip_tex(title.rsplit("}", 1)[0])
    title = " ".join(title.split())

    abstract = src.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    abstract = strip_tex(abstract)
    paras = [" ".join(p.split()) for p in abstract.strip().split("\n\n")
             if p.strip()]
    abstract = "\n\n".join(paras)

    out = os.path.join(ASPLOS, "HOTCRP_fields.txt")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write("TITLE\n=====\n" + title + "\n\n")
        fh.write("ABSTRACT\n========\n" + abstract + "\n")

    print("TITLE")
    print("=====")
    print(title)
    print(f"  ({len(title)} characters)\n")
    print("ABSTRACT")
    print("========")
    print(abstract)
    print(f"\n  ({len(abstract)} characters, {len(abstract.split())} words)")
    print(f"\nwritten to {os.path.relpath(out, os.path.dirname(ASPLOS))}")


if __name__ == "__main__":
    main()
