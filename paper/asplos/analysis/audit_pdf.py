#!/usr/bin/env python3
"""Anonymity audit of the compiled PDFs themselves.

Checking the .tex is not enough: a PDF also carries XMP metadata, an /Info
dictionary, hyperref named destinations (which are built from BibTeX citation
keys), outline titles and compressed object streams.  This decompresses
everything it can and searches the lot for identifying strings.

    python paper/asplos/analysis/audit_pdf.py
"""
from __future__ import annotations

import os
import re
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ASPLOS = os.path.dirname(HERE)

PDFS = ["asplos.pdf", "changes_note.pdf"]

FORBIDDEN = [
    "Chimurkar", "Rishihood", "Newton School", "omchimurkar",
    "Omc12", "CRBench", "Differential-KV", "zenodo", "10.5281", "huggingface",
    "reference removed", "removed for blind", "withheld for",
]

# Metadata fields that must be empty or non-identifying.
META = (b"Author", b"Title", b"Subject", b"Keywords", b"Creator")


def decompressed_blobs(data):
    """Every stream we can inflate, plus the raw file itself."""
    yield data
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        blob = m.group(1)
        try:
            yield zlib.decompress(blob)
        except Exception:
            yield blob


def main():
    failed = False
    for name in PDFS:
        path = os.path.join(ASPLOS, name)
        if not os.path.exists(path):
            print(f"{name}: MISSING")
            failed = True
            continue
        data = open(path, "rb").read()

        haystack = b"\n".join(decompressed_blobs(data)).decode("latin-1")
        hits = sorted({w for w in FORBIDDEN
                       if re.search(re.escape(w), haystack, re.I)})

        # /Info dictionary entries that name a person.
        info = []
        for key in META:
            for m in re.finditer(rb"/" + key + rb"\s*\(([^)]{0,200})\)", data):
                val = m.group(1).decode("latin-1").strip()
                if val:
                    info.append(f"/{key.decode()} = {val!r}")

        print(f"{name}: {len(data):,} bytes")
        print(f"  identifying strings : {hits if hits else 'none'}")
        print(f"  /Info entries       : {info if info else 'none'}")
        if hits:
            failed = True

    print()
    print("PDF ANONYMITY AUDIT: " + ("FAILED" if failed else "clean"))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
