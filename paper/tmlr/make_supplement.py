#!/usr/bin/env python3
"""Build the anonymised code-and-data supplement for the TMLR submission.

TMLR reviews double-blind and requires supplementary material to be anonymised.
The working repository is not: it carries the authors' GitHub handle in the
upstream-repository URLs recorded in every results manifest, in the adapter
bridge, in REPRODUCIBILITY.md, and a given name in the LICENSE copyright line.

This script copies the release tree into a staging directory, rewrites those
strings, refuses to produce an archive if anything identifying survives, and
zips the result.  The working tree is never modified.

Usage:  python paper/tmlr/make_supplement.py [-o OUTPUT.zip]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))

# What ships in the supplement.
INCLUDE_DIRS = ["crbench", "benchmarks", "configs", "examples", "results",
                "scripts", "tests"]
INCLUDE_FILES = ["README.md", "REPRODUCIBILITY.md", "LICENSE", "pyproject.toml"]
# Analysis scripts that regenerate every number and figure in the manuscript.
INCLUDE_ANALYSIS = ("paper/tmlr/analysis", "paper/analysis")

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", ".venv",
                     "node_modules", "third_party", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log", ".zip", ".exe")

# Literal rewrites applied to every text file that ships.
REWRITES = [
    ("https://github.com/Omc12/Differential-KV.git",
     "https://github.com/ANONYMISED/Differential-KV.git"),
    ("https://github.com/Omc12/Differential-KV",
     "https://github.com/ANONYMISED/Differential-KV"),
    ("Omc12/Differential-KV", "ANONYMISED/Differential-KV"),
    ("Copyright (c) 2026 Om", "Copyright (c) 2026 Anonymous Authors"),
]

# Anything matching these must not survive into the archive.
FORBIDDEN = re.compile(r"omchimurkar|chimurkar|Omc12|github\.com/Om\b", re.I)

TEXT_SUFFIXES = (".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
                 ".toml", ".cfg", ".ini", ".rst", "")


def is_text(path):
    return path.endswith(TEXT_SUFFIXES) or os.path.basename(path) == "LICENSE"


def copy_tree(src, dst):
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        for name in files:
            if name.endswith(EXCLUDE_SUFFIXES):
                continue
            s = os.path.join(root, name)
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def scrub(stage):
    """Apply the rewrites in place. Returns the number of files changed."""
    changed = 0
    for root, dirs, files in os.walk(stage):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        for name in files:
            path = os.path.join(root, name)
            if not is_text(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, OSError):
                continue
            new = text
            for old, repl in REWRITES:
                new = new.replace(old, repl)
            if new != text:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
                changed += 1
    return changed


def verify(stage):
    """Fail loudly rather than ship a deanonymising archive."""
    hits = []
    for root, dirs, files in os.walk(stage):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, stage)
            if FORBIDDEN.search(rel):
                hits.append((rel, 0, "path"))
            if not is_text(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        m = FORBIDDEN.search(line)
                        if m:
                            hits.append((rel, i, line.strip()[:120]))
            except (UnicodeDecodeError, OSError):
                continue
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output",
                    default=os.path.join(_HERE, "crbench_supplement_anon.zip"))
    args = ap.parse_args()

    stage = tempfile.mkdtemp(prefix="crbench_supp_")
    try:
        for d in INCLUDE_DIRS:
            src = os.path.join(REPO, d)
            if os.path.isdir(src):
                copy_tree(src, os.path.join(stage, d))
        for f in INCLUDE_FILES:
            src = os.path.join(REPO, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(stage, f))
        for rel in INCLUDE_ANALYSIS:
            src = os.path.join(REPO, *rel.split("/"))
            if os.path.isdir(src):
                copy_tree(src, os.path.join(stage, "paper", "analysis"))
                break

        changed = scrub(stage)
        hits = verify(stage)
        if hits:
            print("REFUSING to build: identifying strings survive scrubbing")
            for rel, line, snippet in hits[:40]:
                print(f"  {rel}:{line}: {snippet}")
            raise SystemExit(1)

        n = 0
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(stage):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
                for name in sorted(files):
                    p = os.path.join(root, name)
                    z.write(p, os.path.relpath(p, stage))
                    n += 1

        size = os.path.getsize(args.output) / 1e6
        print(f"scrubbed {changed} file(s); anonymisation check passed")
        print(f"wrote {args.output}  ({n} files, {size:.1f} MB)")
        if size > 100:
            print("WARNING: TMLR caps supplementary material at 100 MB")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
