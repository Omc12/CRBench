#!/usr/bin/env python3
"""Verify that every bibliography entry lists its authors in full.

ACM's reference format lists all authors; the ASPLOS submission therefore
carries no "et al." in the bibliography.  This script proves that claim rather
than asserting it: for every entry with a known arXiv identifier it fetches the
canonical author list and compares surnames, in order, against refs.bib.

Entries with no arXiv record (USENIX-only, ACM-DL-only, a corporate author, or
the de-identified self-citation) are listed with the source they were checked
against by hand.

    python paper/asplos/analysis/verify_authors.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET

BIB = "../refs.bib"

# arXiv identifiers, each taken from the published record of that exact paper.
ARXIV = {
    "kwon2023pagedattention": "2309.06180",
    "zhong2024distserve": "2401.09670",
    "agrawal2024sarathi": "2403.02310",
    "dao2022flashattention": "2205.14135",
    "liu2024cachegen": "2310.07240",
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
    "agrawal2026benchmarking": "2607.05399",
}

# Entries with no arXiv record, and the source each was checked against.
OFF_ARXIV = {
    "yu2022orca":
        "usenix.org/conference/osdi22/presentation/yu -- the page's own BibTeX "
        "gives exactly: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, "
        "Soojeong Kim, Byung-Gon Chun",
    "fan2026zipserv":
        "ACM DL record 10.1145/3779212.3790250 -- all 8 authors listed",
    "kamradt2023needle":
        "single-author software artifact (Greg Kamradt); repository resolves",
    "reid2024gemini":
        "arXiv:2403.05530 lists 1,137 authors headed by the corporate author "
        "'Gemini Team'; the corporate author is cited, as the record credits",
    "dkvpreprint":
        "de-identified for double-blind review; see verify_citations.py",
}


def curl(url):
    out = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "40",
         "-H", "User-Agent: crbench-author-verify/1.0", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:150])
    return out.stdout


def fold(s):
    """Compare names without accents, case or punctuation."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def bib_authors(field):
    """Surnames from a BibTeX author field, in order."""
    out = []
    for a in field.split(" and "):
        a = a.strip()
        if not a:
            continue
        if a.startswith("{{") or a.startswith("{"):        # corporate author
            out.append(fold(a.strip("{}")))
        elif "," in a:
            out.append(fold(a.split(",")[0]))
        else:
            out.append(fold(a.split()[-1]))
    return out


def arxiv_surnames(aid):
    root = ET.fromstring(curl(
        "http://export.arxiv.org/api/query?id_list=" + aid + "&max_results=1"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    if e is None:
        return None
    names = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
    return [fold(n.split()[-1]) for n in names if n.strip()]


def entries(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(2).strip(), m.group(3)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            f[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out.append((key, f))
    return out


def main():
    ents = entries(BIB)
    print(f"checking author lists for {len(ents)} entries\n")
    bad = []

    for key, f in ents:
        author = f.get("author", "")
        if re.search(r"\bet al\b|and others", author, re.I):
            print(f"  FAIL   {key:<26} still abbreviates the author list")
            bad.append(key)
            continue

        if key in OFF_ARXIV:
            print(f"  manual {key:<26} {OFF_ARXIV[key]}")
            continue

        aid = ARXIV.get(key)
        if not aid:
            print(f"  CHECK  {key:<26} no source registered for this entry")
            bad.append(key)
            continue

        try:
            canon = arxiv_surnames(aid)
        except Exception as exc:
            print(f"  ERROR  {key:<26} arXiv lookup failed: {exc}")
            bad.append(key)
            continue

        ours = bib_authors(author)
        if ours == canon:
            print(f"  ok     {key:<26} {len(ours)} authors, order matches "
                  f"arXiv:{aid}")
        else:
            print(f"  FAIL   {key:<26} bib has {len(ours)}, arXiv:{aid} has "
                  f"{len(canon)}")
            print(f"           bib   : {ours}")
            print(f"           arXiv : {canon}")
            bad.append(key)
        time.sleep(1.2)

    print()
    if bad:
        print(f"{len(bad)} entr{'y' if len(bad) == 1 else 'ies'} need "
              f"attention: {bad}")
        sys.exit(1)
    print("every entry lists its authors in full, in the canonical order")


if __name__ == "__main__":
    main()
