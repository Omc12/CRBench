#!/usr/bin/env python3
"""Fetch complete author lists for the refs.bib entries that still say
"and others", so the bibliography can list every author as ACM's reference
format requires -- without any name being reconstructed from memory.

For each entry it prints the full list from arXiv (by identifier, when the
entry has one) and from OpenAlex (by title). Where the two agree the list is
safe to paste in; where they disagree or a source is missing, the entry is
flagged for a human decision rather than guessed.

    python paper/asplos/analysis/fetch_authors.py
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

BIB = "../refs.bib"

# arXiv identifiers for entries whose bib record does not carry one in a
# machine-readable field. Every id here was taken from the published record of
# that exact paper, not inferred.
ARXIV = {
    "kwon2023pagedattention": "2309.06180",
    "zhong2024distserve": "2401.09670",
    "agrawal2024sarathi": "2403.02310",
    "dao2022flashattention": "2205.14135",
    "liu2024kivi": "2402.02750",
    "li2024snapkv": "2404.14469",
    "xiao2023streamingllm": "2309.17453",
    "zhang2023h2o": "2306.14048",
    "liu2023scissorhands": "2305.17118",
    "hooper2024kvquant": "2401.18079",
    "dettmers2022llmint8": "2208.07339",
    "ge2023model": "2310.01801",
    "zhang2024pyramidkv": "2406.02069",
    "sharma2023truth": "2312.13558",
    "hsieh2024ruler": "2404.06654",
    "bai2023longbench": "2308.14508",
    "an2023leval": "2307.11088",
    "zhang2024infinitebench": "2402.13718",
    "reid2024gemini": "2403.05530",
    "liu2024cachegen": "2310.07240",
    "fan2026zipserv": None,          # ACM DL only
    "yu2022orca": None,              # USENIX only
}


def curl(url):
    out = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "40",
         "-H", "User-Agent: crbench-author-fetch/1.0", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:150])
    return out.stdout


def clean(t):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())


def arxiv_authors(aid):
    root = ET.fromstring(curl(
        "http://export.arxiv.org/api/query?id_list=" + aid + "&max_results=1"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    if e is None:
        return None, None
    title = " ".join((e.findtext("a:title", "", ns) or "").split())
    names = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
    return title, names


def openalex_authors(title):
    for url in ("https://api.openalex.org/works?filter=title.search:"
                + urllib.parse.quote(clean(title)) + "&per-page=5",
                "https://api.openalex.org/works?search="
                + urllib.parse.quote(clean(title)) + "&per-page=5"):
        try:
            res = json.loads(curl(url)).get("results", [])
        except Exception:
            res = []
        if res:
            best = max(res, key=lambda w: difflib.SequenceMatcher(
                None, clean(title), clean(w.get("title") or "")).ratio())
            score = difflib.SequenceMatcher(
                None, clean(title), clean(best.get("title") or "")).ratio()
            names = [(a.get("author") or {}).get("display_name", "")
                     for a in best.get("authorships", [])]
            return best.get("title"), names, score
        time.sleep(1.5)
    return None, None, 0.0


def bib_entries(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(2).strip(), m.group(3)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            f[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out.append((key, f))
    return out


def surnames(names):
    return [n.split()[-1].lower() for n in names if n]


def main():
    for key, f in bib_entries(BIB):
        if "others" not in f.get("author", ""):
            continue
        title = re.sub(r"[{}]", "", f.get("title", ""))
        print("=" * 78)
        print(f"{key}\n  bib title: {title}")

        ax_names = None
        aid = ARXIV.get(key)
        if aid:
            try:
                ax_title, ax_names = arxiv_authors(aid)
                sim = difflib.SequenceMatcher(None, clean(title),
                                              clean(ax_title)).ratio()
                print(f"  arXiv:{aid} (title match {sim:.2f}) -> {ax_title}")
                print(f"    {len(ax_names)} authors: {'; '.join(ax_names)}")
                if sim < 0.8:
                    print("    !! TITLE MISMATCH -- do not use without checking")
            except Exception as exc:
                print(f"  arXiv lookup failed: {exc}")
            time.sleep(1.0)

        oa_title, oa_names, oa_score = openalex_authors(title)
        if oa_names:
            print(f"  OpenAlex (title match {oa_score:.2f}) -> {oa_title}")
            print(f"    {len(oa_names)} authors: {'; '.join(oa_names)}")
        else:
            print("  OpenAlex: no hit")

        if ax_names and oa_names:
            agree = surnames(ax_names) == surnames(oa_names)
            print(f"  >> sources {'AGREE' if agree else 'DISAGREE'} "
                  f"({len(ax_names)} vs {len(oa_names)} authors)")
        time.sleep(1.5)


if __name__ == "__main__":
    main()
