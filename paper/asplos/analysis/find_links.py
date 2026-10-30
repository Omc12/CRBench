#!/usr/bin/env python3
"""Find an authoritative, verifiable link for every bibliography entry.

Preference order, as ASPLOS/ACM expect: a DOI, then the official proceedings
page, then the arXiv abstract page.  Nothing is emitted that was not confirmed
by an actual request: a Crossref DOI must match the title closely and agree on
year, a proceedings URL must return 200 and contain the title, and an arXiv id
must resolve to a matching title.

Prints a report; it does not edit refs.bib.

    python paper/asplos/analysis/find_links.py
"""
from __future__ import annotations

import difflib
import io
import json
import re
import subprocess
import time
import urllib.parse
import sys
import xml.etree.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                              encoding='utf-8', errors='replace')

BIB = "../refs.bib"

# Publisher prefixes whose DOI is the canonical record for the work:
# ACM, ACL Anthology, IEEE, Springer. Deliberately excluded: 10.52202
# (Curran reprints of NeurIPS/ICML volumes) -- those papers are cited to
# their open-access record instead.
CANONICAL_PREFIXES = ("10.1145", "10.18653", "10.1109", "10.1007")

# Verified arXiv identifiers (each confirmed against the record in this repo's
# verify_authors.py run: title and full author list matched).
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
    "reid2024gemini": "2403.05530",
    "agrawal2026benchmarking": "2607.05399",
}

# Candidate official proceedings pages to test (USENIX has no DOIs).
PROCEEDINGS = {
    "yu2022orca": "https://www.usenix.org/conference/osdi22/presentation/yu",
    "zhong2024distserve":
        "https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin",
    "agrawal2024sarathi":
        "https://www.usenix.org/conference/osdi24/presentation/agrawal",
}

# Entries that must not carry a link.
NO_LINK = {
    "dkvpreprint": "de-identified self-citation; any link deanonymises it",
}


def curl(url, head=False):
    cmd = ["curl", "-sS", "-L", "--max-time", "40",
           "-H", "User-Agent: crbench-link-check/1.0"]
    if head:
        cmd += ["-o", "-", "-w", "\n__STATUS__%{http_code}"]
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:120])
    return out.stdout


def clean(t):
    t = re.sub(r"[{}$\\]", "", t or "")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split())


def sim(a, b):
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def crossref_doi(title, year):
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(clean(title)) + "&rows=3")
    try:
        items = json.loads(curl(url))["message"]["items"]
    except Exception:
        return None
    for it in items:
        cand = (it.get("title") or [""])[0]
        s = sim(title, cand)
        parts = (it.get("issued", {}).get("date-parts") or [[None]])[0]
        yr = str(parts[0]) if parts and parts[0] else ""
        doi = it.get("DOI", "")
        canonical = any(doi.startswith(p) for p in CANONICAL_PREFIXES)
        if (s >= 0.90 and canonical
                and (not year or not yr or abs(int(yr) - int(year)) <= 1)):
            return doi, cand, yr, s
    return None


def arxiv_ok(aid, title):
    root = ET.fromstring(curl(
        "http://export.arxiv.org/api/query?id_list=" + aid + "&max_results=1"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    if e is None:
        return None
    t = " ".join((e.findtext("a:title", "", ns) or "").split())
    return sim(title, t), t


def url_ok(url, title):
    body = curl(url, head=True)
    m = re.search(r"__STATUS__(\d+)\s*$", body)
    code = m.group(1) if m else "?"
    has_title = clean(title)[:45] in clean(body)
    return code, has_title


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
    print(f"establishing links for {len(ents)} entries\n")
    plan = {}

    for key, f in ents:
        title = re.sub(r"[{}]", "", f.get("title", ""))
        year = f.get("year", "")
        print(f"--- {key}")

        if key in NO_LINK:
            print(f"    SKIP: {NO_LINK[key]}")
            plan[key] = None
            continue

        if f.get("doi"):
            print(f"    already has DOI {f['doi']}")
            plan[key] = ("doi", f["doi"])
            continue
        if "url{" in f.get("howpublished", ""):
            print(f"    already has URL in howpublished")
            plan[key] = ("keep", f["howpublished"])
            continue

        # 1. DOI via Crossref
        got = crossref_doi(title, year)
        time.sleep(1.0)
        if got:
            doi, cand, yr, s = got
            print(f"    DOI {doi}  (title~{s:.2f}, year {yr})")
            print(f"        -> {cand[:80]}")
            plan[key] = ("doi", doi)
            continue
        print("    no confident Crossref DOI")

        # 2. Official proceedings page
        if key in PROCEEDINGS:
            try:
                code, ok = url_ok(PROCEEDINGS[key], title)
                print(f"    proceedings {PROCEEDINGS[key]} -> HTTP {code}, "
                      f"title present: {ok}")
                if code == "200" and ok:
                    plan[key] = ("url", PROCEEDINGS[key])
                    time.sleep(1.0)
                    continue
            except Exception as exc:
                print(f"    proceedings check failed: {exc}")
            time.sleep(1.0)

        # 3. arXiv
        aid = ARXIV.get(key)
        if aid:
            try:
                s, t = arxiv_ok(aid, title)
                print(f"    arXiv:{aid} (title~{s:.2f}) -> {t[:70]}")
                if s >= 0.85:
                    plan[key] = ("url", f"https://arxiv.org/abs/{aid}")
                    time.sleep(1.0)
                    continue
            except Exception as exc:
                print(f"    arXiv check failed: {exc}")
            time.sleep(1.0)

        print("    !! NO RELIABLE LINK ESTABLISHED")
        plan[key] = None

    print("\n" + "=" * 70)
    print("PLAN")
    for key, v in plan.items():
        print(f"  {key:<26} {v}")


if __name__ == "__main__":
    main()
