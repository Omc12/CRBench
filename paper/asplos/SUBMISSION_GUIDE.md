# ASPLOS 2027 (September cycle) — submission guide

Working notes for the repositioned manuscript in this directory. Everything here
refers to files under `paper/asplos/`.

| File | What it is |
|---|---|
| `asplos.tex` / `asplos.pdf` | The submission. Body ends on **page 11**; refs + appendices follow. |
| `changes_note.tex` / `changes_note.pdf` | The one-page "changes since previous submissions" note the CFP requires as a separate upload. |
| `refs.bib` | 25 references, every one checked against an external record. |
| `figures/`, `tables/` | Regenerated at ASPLOS printed sizes from the existing study data. |
| `analysis/` | `make_figures.py`, `build_tables.py`, `verify_claims.py`, `verify_citations.py`. |

**Deadline: the September-cycle full-paper deadline is 9 September 2026, AoE.**

## What "TriAxis" is

It is a **placeholder name for your harness, required by the CFP**, not a new
thing. Your harness is public as *CRBench* at a repository that carries your
name, and the CFP says: *"If your system is already released to the public,
rename it in your submission."* Leaving it as CRBench would let any reviewer
search the name and find you.

So the paper calls it **TriAxis** (after the three-axis framing). It is defined
once, in `asplos.tex`:

```latex
\newcommand{\sys}{TriAxis}
```

and used in six places. Change that one line to rename it. Change it back to
CRBench in the camera-ready, once double-blind no longer applies.

## Bibliography link pass

Every reference now carries a clickable, verified link, except the one that
must not have one. Nothing was constructed by hand: `analysis/find_links.py`
resolves each entry and only emits a link it confirmed by request — a Crossref
DOI had to match the title to ≥0.90 and agree on year, a proceedings URL had to
return HTTP 200 *and* contain the title, and an arXiv id had to resolve to a
matching title.

**21 links added**, in the CFP's preference order:

| Source | Count | Entries |
|---|---|---|
| Publisher DOI (`10.1145`, `10.18653`) | 4 added + 2 already present | PagedAttention, LongBench, L-Eval, ∞-Bench; ZipServ and CacheGen already had theirs |
| Official proceedings page (USENIX — no DOI exists) | 3 | Orca, DistServe, Sarathi-Serve |
| arXiv abstract page | 14 | the NeurIPS/ICML/ICLR papers and the arXiv-only entries |
| Already present | 1 | Needle-in-a-Haystack (third-party repository) |
| **Deliberately none** | **1** | **the DKV self-citation** |

One judgement call worth knowing: Crossref returns `10.52202/...` DOIs for the
NeurIPS papers. Those are **Curran's printed-proceedings reprints**, not the
open-access record a reader wants, so `find_links.py` restricts DOIs to
canonical publisher prefixes and those entries fall through to arXiv instead.
Change `CANONICAL_PREFIXES` in that script if you disagree.

**The DKV entry deliberately has no link.** Its only stable link is the Zenodo
DOI, which resolves to a page naming an author of this paper. It renders as:

> [9] DKV Project. 2026. DKV: Anchor + Low-Rank Differential KV-Cache
> Compression for Scalable Long-Context Inference. Preprint.

All 24 URLs that reach the PDF are institutional (arxiv.org, doi.org,
usenix.org) plus one third-party repository (gkamradt's Needle-in-a-Haystack).
None points to an author account.

## Final compliance pass (bibliography + self-citation)

Scope was limited to compliance; no science, no figures, no tables, no
equations, and **no change to pages 1–2** (verified: every intro marker sits on
the same page as before, and Figure 1 is still at the top of page 2).

**Self-citation.** The DKV reference previously printed a personal author name
and a Zenodo DOI that resolves to a page naming that author. It now reads:

> DKV Project. 2026. DKV: Anchor + Low-Rank Differential KV-Cache Compression
> for Scalable Long-Context Inference. Preprint.

No personal name, no identity-resolving link, and no redaction marker of the
kind the CFP forbids ("reference removed for blind review"). The BibTeX key was
also renamed `dkv2026zenodo` → `dkvpreprint`, because hyperref embeds citation
keys as PDF named destinations and the old key contained "zenodo".

*Be aware of the trade-off:* the CFP's own prescription for one's public prior
work is the opposite — cite it in the third person **with** the real name. The
de-identified form is the stronger anonymity choice but is a small liberty with
attribution, and it does not fully anonymise the fact that DKV is evaluated,
since the method name appears throughout the figures and tables. Restore the
full record in the camera-ready, and disclose the relationship to the chairs.

**Bibliography.** Every "and others"/"et al." is gone: all 25 entries now list
their complete author list. The lists were fetched from arXiv, USENIX and the
ACM DL — none was written from memory. `verify_authors.py` re-checks 20 of them
against arXiv surname-by-surname and in order, and records the source for the
five with no arXiv record. Two errors surfaced while doing this:

- **PyramidKV's first author was wrong.** The entry began "Zhang, Yichi"; the
  record's first author is **Zefan Cai**. Full 11-author list now correct.
- **Gemini 1.5** was cited as "Reid, Savinov, Teplyashin and others". The record
  lists **1,137 authors** headed by the corporate author **"Gemini Team"**,
  which is now what is cited.

## Corrections made during the final audit

These were real errors, found by reading the manuscript against the data rather
than by any check that existed at the time. All are fixed, and all are now
covered by `verify_claims.py` (which grew from 49 to 64 checks).

| Was | Should have been | Where |
|---|---|---|
| metadata "6–28%" | **6–23%** — no operating point reaches 28%; the real span of the metadata share is 5.8–23.2% | abstract, contributions, §5.2, §6, §9 |
| factorised state "18–23%" | **10–23%** — DKV-High on Qwen2.5 7B is 10.4%, below the stated floor | §1, Fig. 3 caption |
| "SnapKV … the third on Qwen2.5 7B" | SnapKV is **second** there; the sentence was replaced with instability examples that are actually true (Low-Rank 3rd→4th, StreamingLLM 5th→3rd at 8 bits) | §5.1 |
| KIVI/SnapKV on Qwen "differ by 11.8×" | **11.7×** — 11.8× is SnapKV's inflation against *its own* dense run, which is a different quantity; the paper now uses each in the right place | §5.3 |
| $\mathcal{M}$ used for both the model and peak memory | peak memory is now $\mathcal{P}$ | §3.1, §3.4 |
| "eviction and pooling methods are untouched" | they are not *charged*, but they can still move rank when a neighbour falls past them | §5.2 |
| changes note quoted the old title containing the public system name | removed, with a sentence explaining why | `changes_note.tex` |

Two other things the audit changed: Table 4 moved to Appendix C to recover the
page budget, and a second figure guard was added that fails generation if any
artist is cropped (it immediately caught a clipped legend and a clipped axis
label that were already in the PDF).

---

## A. Why the previous version was not yet an ASPLOS paper

Nothing in the v2 manuscript was wrong. The problem was that it was organised as
a *benchmark paper* and ASPLOS reads *systems* papers. Concretely:

1. **The framework was the subject, the findings were the evidence.** v2's arc was
   "here is CRBench → here is what it measures → here is a table." An ASPLOS PC
   member wants "here is a systems problem → here is a surprising measurement →
   here is what it changes." The findings were already the strongest part of the
   paper and were structurally demoted.
2. **The systems vocabulary was missing.** TTFT, peak VRAM and decode throughput
   were reported, but never connected to the decisions a serving system makes —
   admission control, capacity planning, latency objectives, prefill/decode
   scheduling. The paper measured serving without talking about serving.
3. **No positioning against the serving-systems literature.** v2's related work
   cited compression methods and long-context benchmarks. It cited no serving
   system. To an ASPLOS reviewer this reads as a paper from an adjacent
   community that has not engaged with theirs.
4. **The first two pages did not carry the contribution.** ASPLOS's rapid round
   reads *only* pages 1–2 and reviewers are instructed not to read past them.
   In v2, page 2 was still in framework exposition.
5. **A real conceptual bug.** The paper argued the axes are independent while
   reporting an S_sys that contains S_res by construction. v2 acknowledged this
   in one paragraph but still led with S_sys in the tables.
6. **Over-strong statistical language.** "None of them predicts the others" is
   not what 36 operating points can establish.
7. **Length and formatting.** Single-column TMLR at 16 pages, with figures whose
   text would have fallen to ~6pt if dropped into a 3.3″ ASPLOS column.
8. **Concurrent work was not addressed.** See the note under section I below —
   this was the most dangerous omission.

## B. Titles considered

1. Beyond Compression Ratio: Measuring the Systems Utility of Long-Context KV-Cache Compression
2. **Compression Ratio Is Not a Serving Metric: A Measurement Study of KV-Cache Compression for Long-Context LLM Serving** ← chosen
3. Three Axes, Two Numbers: Why KV-Cache Compression Reporting Misleads Serving Systems
4. What a Compression Ratio Does Not Tell You: Paired Measurement of KV-Cache Compression for Long-Context Serving
5. The Missing Axis: Representation Footprint, Task Utility, and Realised Serving Cost in KV-Cache Compression

**Why 2.** It states a falsifiable systems claim in the first five words, names the
paper as a measurement study (which the CFP explicitly welcomes), and puts
"serving" in the title so the rapid round can place it. (1) buries the claim
behind "beyond"; (3) is memorable but reads as an opinion piece; (4) and (5) lead
with methodology.

## C–E. Abstract, first two pages, contributions

Rewritten in `asplos.tex`. The measured result of the restructuring:

| What a rapid reviewer sees | Page |
|---|---|
| Problem, reporting convention, the question | 1 |
| Finding 1 (18.7× accuracy spread at matched bitrate) | 1 |
| Finding 2 (30.3× TTFT at matched bitrate *and* accuracy) | 1–2 |
| Figure 1 (the three-panel anchor) | 2 |
| Finding 3 (metadata understatement, 6–23%) | 2 |
| "Why this matters for serving systems" | 2 |
| Contributions heading + lead-in (what the methodology adds) | 2 |
| The four contribution bullets | 3 |
| Scope limits + concurrent work | 3 |

By the end of page 2 a rapid reviewer has the systems problem, why conventional
reporting is insufficient for it, the three quantitative findings, their
consequences for capacity planning and latency objectives, and the prose
statement of what the measurement design contributes. The four explicit
contribution bullets begin immediately on page 3. Verified from label positions
in `asplos.aux`, not by eye.

Contribution order is now: (1) empirical characterisation, (2) the measurement
methodology that made it possible, (3) honest memory accounting, (4) provenance
and reproducibility. The composite scores are no longer the apparent novelty.

## F. What was cut, merged, or moved

| Change | Effect |
|---|---|
| §"Three axes" compressed ~40% — the axis definitions no longer restate the intro | −0.4 p |
| §5.4 (task structure, context scaling) reduced to two paragraphs; both its figures moved to Appendix C | −0.45 p |
| Determinism paragraph → Appendix D (a one-sentence summary stays in §4) | −0.12 p |
| Part-1 aggregation detail (byte-weighting, failed-query handling) → Appendix D | −0.1 p |
| Threats section tightened item by item | −0.18 p |
| Provenance table notes column tightened | −0.1 p |
| Related work concurrent-work paragraph tightened | −0.12 p |
| Model-dependence discussion deduplicated between §1 and §5.3 | −0.15 p |
| Separate "what we had to build" paragraph merged into the contributions lead-in | −0.14 p |

**Kept in the body deliberately:** the full 39-row three-axis table (Table 2), the
provenance table, the matched-budget table, and all four systems figures. The
exhaustive table was moved to the appendix mid-edit and then moved *back* once
the text cuts created room — for a measurement paper it is the evidence.

**Moved to appendices (which do not count against the limit):** scoring proofs
(A), alternative formulations and α-sensitivity (B), per-task/per-context/
per-operating-point figures (C), measurement protocol (D), artifact appendix (E).

## G. Final structure, as measured

| § | Section | Starts | Budget |
|---|---|---|---|
| — | Title, abstract | 1 | 0.3 p |
| 1 | Introduction (incl. Fig. 1) | 1 | 2.2 p |
| 2 | Three axes, and why one does not determine another | 3 | 0.5 p |
| 3 | TriAxis: measuring the three axes on one query (Fig. 2) | 3 | 1.9 p |
| 4 | Study design + provenance (Table 1) | 5 | 1.4 p |
| 5 | Results (Tables 2–3, Figs. 3–4) | 6 | 3.7 p |
| 6 | Implications for serving systems | 9 | 0.6 p |
| 7 | Threats to validity and scope | 10 | 0.7 p |
| 8 | Related work (Table 4) | 11 | 0.7 p |
| 9 | Conclusion | 11 | 0.2 p |
| — | Generative AI disclosure | 11 | excluded from limit |
| — | References | 12 | excluded |
| A–B | Appendices (proofs, formulations) | 12 | excluded |
| C–E | Appendices (detail, protocol, artifact) | 13 | excluded |
| — | Appendix figures | 14–15 | excluded |

Total document: 15 pages.

**Body ends on page 11.** Re-check after any edit with:
`grep -a "newlabel{concend}" asplos.aux` → the second `{}` group is the page.

## H. The S_res / S_sys fix

The conceptual problem and how it is now handled (no formula and no measured
value changed):

- **S_res** — representation + task utility. Unchanged.
- **φ** — the *pure* hardware term, built only from paired TTFT, throughput and
  peak-memory ratios. **It contains no term from Part 1.** This is now the
  serving axis, and every independence claim in the paper is made against it.
- **S_sys** = α·S_res + (1−α)·100φ — retained only as a single-column
  convenience for readers who want one ordering. §3.4 states explicitly that it
  is *not* evidence, that a correlation between S_sys and S_res is circular by
  construction, and that **nothing in the paper's argument would change if S_sys
  were deleted.**
- One consequence is now stated where it matters: the S_res→S_sys rank
  displacements (1.4 / 2.6 / 1.4 positions) are a **lower bound** on the
  disagreement between the axes, precisely because the two orderings share the
  S_res term.
- Table 2's caption labels which column carries evidence and which does not.

## I. Claims flagged and how they were handled

| Original claim | Status |
|---|---|
| "none of them predicts the others" (conclusion) | → "none of them **reliably determines** the others **in our measurements**" |
| "the memory a method saves carries essentially no information about…" | → "we **detect no** rank relationship" |
| "the association is absent rather than merely weak" | → "not merely weak; **not detectable at this sample size**" |
| "A compression ratio reported without a task measurement is close to uninformative" | → "**in these measurements**, a bitrate is not sufficient to order these methods" |
| Fig. 1(c) panel title "Footprint says nothing about prefill cost" | → "Footprint **does not track** prefill cost" (regenerated) |
| §5.1 "A capacity planner who picks the method that frees the most cache is not thereby picking a method that answers the question" | Cut — rhetorical, not measured |
| p-values | Now explicitly quoted "**only to indicate that no relationship was detectable, never to assert that none exists**" |

**Deliberately left strong,** because they are direct matched observations rather
than inferences: the 18.7× accuracy spread at matched bitrate, the 30.3× TTFT
and 3.46× VRAM gap between two methods matched on both reported numbers, the
6–28% metadata correction, and the model-dependence (30.3× / 11.8× / 1.04×).

**Two items still worth your judgement:**

1. **The title** asserts "Compression Ratio Is Not a Serving Metric." It is
   supported for the measured regime, and the abstract and §7 scope it. If a
   reviewer pushes back, "…Is Not Enough" is the fallback.
2. **Concurrent work.** `arXiv:2607.05399` (Agrawal & Mayer) benchmarks
   KV-cache optimisations on task quality *and* system performance and already
   reports that "compression ratio alone is a poor predictor of end-to-end
   performance." This is close to the paper's headline. It is now cited in §1
   and given a dedicated paragraph in §8 stating exactly what this paper's
   measurement design adds (query-level pairing, metadata-charged accounting,
   provenance/realisation, model-dependence). **Do not remove this.** A reviewer
   who knows that paper and does not see it cited will treat the submission as
   scooped.

## J. Captions rewritten to state the takeaway

| Figure/Table | Caption now leads with |
|---|---|
| Fig. 1 | "The two numbers a compression paper reports do not determine what a serving system pays." |
| Fig. 2 | "Every measurement in this paper is a difference between two executions of the same prompt." |
| Fig. 3 | "The bits a method is charged for and the bits it actually needs are not the same number, and the gap is family-specific." |
| Fig. 4 | "The serving cost of a representation is a property of the deployment, not of the representation." |
| Fig. 5 | "One of our five tasks measures the base model, not the representation." |
| Fig. 6 | "The cliff beyond 64K belongs to the model, not to compression." |
| Table 1 | "A serving number is a property of a code path, so we say which one." |
| Table 2 | "Reading across a row shows the three axes disagreeing." |
| Table 3 | "Identical budget, three models, and no stable ordering." |
| Table 4 | "The disagreement between axes is invisible unless the axes are measured on the same query." |

## L. Compliance checklist

The CFP's page rule, verbatim:

> "Full submissions must not exceed 11 pages of single-spaced two-column text."
> "This page limit applies to all text, figures, tables, and footnotes."
> "The only exceptions are the acknowledgment section (used only to acknowledge
> use of Generative AI as per ACM policy above), the bibliographic references
> section, and the appendices, which are not included in the page limit."

So: **11 pages** covering text, figures, tables and footnotes; the AI-disclosure
acknowledgment, the references and the appendices are all excluded. (Accepted
papers get +2 pages in the camera-ready, and may buy 1–2 more — not relevant at
submission.)

| Requirement (CFP wording) | Status |
|---|---|
| "must not exceed 11 pages of single-spaced two-column text" | **Body ends on page 11.** Conclusion and the AI disclosure both land on p11, and the disclosure is itself excluded, so the counted content is ≤11 pages. |
| References excluded, "space they occupy should not be optimized away" | References start p12, full ACM-Reference-Format, nothing compressed. |
| Appendices "do not count towards the page limit" | Appendices A–E, pp. 12–15. |
| "Supplemental material and Appendixes may be submitted **as part of the submission file** and have no page limit" | One PDF, 15 pages total. **No separate supplementary upload.** |
| "The submission must be self-contained within the page limit, and the supplemental material/appendixes should not be used as a way to circumvent this limit" | Checked. Every claim in the paper is stated with its numbers inside pp. 1–11: all three findings, the full per-operating-point table, the provenance table, the matched-budget table, the metadata figure and the φ figure are all in the body. The appendices hold only proofs, alternative formulations, α-sensitivity, four supporting figures, protocol detail and the artifact appendix. The §5.4 results are argued in body prose with their numbers; only their figures sit in Appendix C. |
| "must use the ACM's acmart Latex class" with "sigplan, anonymous, review, and non-acm options" | `\documentclass[sigplan,10pt,anonymous,review,nonacm]{acmart}` — exactly. |
| "text must use a 10pt font (not 9pt) or bigger" | Verified by probing `\f@size`: **body 10pt**, baselineskip 12pt. |
| Figure/table text "9pt or bigger", minimum 8pt | Tables at `\small` = **9pt**. Figures generated at final printed size (scale 1.0) with two assertions in `figstyle.py` that **fail generation**: one if any figure text falls below 8pt, one if any artist extends past the canvas (which, because we save without a tight bounding box, would silently crop it). See the note below. |
| "Squeezing space is forbidden" | No vertical-space, font-size or margin manipulation. Two knobs are set, neither of which fits more content on a page: `\emergencystretch=1.2em` (horizontal inter-word stretch on lines that would otherwise overflow the narrow column) and a block of float-placement parameters **declared after `\appendix`**, so they cannot affect the 11-page body — they only let the float-heavy, text-light appendix pack figures onto shared pages instead of stranding most of a page behind each one. Largest remaining overfull box is under 10pt. |
| Double-blind anonymity | No author names, affiliations, emails or repository URLs. `verify_claims.py` greps for every identifying string from the public artifacts and passes. PDF XMP `dc:creator` and `dc:title` are empty. |
| "If your system is already released to the public, rename it" | The harness is public under its real name, so it appears throughout as **TriAxis** via `\newcommand{\sys}`. Change the name in one line if you prefer another. |
| "Do not include a 'reference removed for blind review' text" | The old `[Authors withheld]` bibliography entry is **gone**. |
| "When it is necessary to cite your own studies, cite them as written by a third party" | DKV is cited normally as `Chimurkar, Zenodo, DOI 10.5281/zenodo.21539110`. All "our own prior work" language removed from the body. The BibTeX key was renamed so the surname does not appear in the `.tex` either. |
| Generative AI disclosure "must be fully disclosed", "immediately before the References" | Unnumbered **"Generative AI Disclosure"** section immediately before the bibliography, stating assistive use for drafting/editing only, that no AI tool designed the study or produced/analysed measurements, and that no AI tool is an author. (Anonymous mode suppresses `\acks`, so a named section is used instead.) |
| Note describing changes since previous submissions, "limit this document to one page" | `changes_note.pdf` — **one page**, anonymous, states plainly that both TMLR submissions were desk-rejected with no reviewer reports and that no change responds to a review. |
| Originality / concurrent submission | Both prior submissions were **desk-rejected, not published**; the CFP states technical reports and arXiv postings are not prior publications. Confirm before submitting that the work is not under review anywhere else. |
| Conflicts of interest | You must enter these in HotCRP: your institution(s) for the past four years, advisors (forever), co-authors from the last four years, funders. |
| Artifact | Appendix E describes scope, contents and reproduction. It claims only what exists: records + four analysis scripts that regenerate every number. **Do not upload an anonymised artifact archive claiming more than that.** |

### Things only you can do

1. **Enter conflicts** in HotCRP.
2. **Declare the DKV relationship to the chairs.** The study evaluates a public
   method authored by one of the authors. Citing it in the third person is what
   the CFP prescribes, and that is what the paper does — but the honest move is
   a note in the HotCRP comments-to-chairs field saying so. The self-disclosure
   paragraph that was in the TMLR version had to come out for anonymity; put it
   back in the camera-ready.
3. **Decide on the pseudonym.** "TriAxis" is a placeholder; one line in the
   preamble.
4. **Consider whether the TMLR submission numbers should stay** in the changes
   note. They are there because you asked for them and they are the transparent
   choice. If you would rather not give reviewers something to search, delete
   the two parenthetical numbers — the note reads fine without them.

### Note on figure font sizes

The CFP asks for figure text that "appears to readers as a 9pt font or bigger"
and forbids anything under 8pt. The figures here use **8pt** for tick labels and
annotations and 8.5–9pt for axis labels and titles — compliant with the stated
minimum, at the low end of the recommendation. Raising the floor to 9pt would
require taller figures, and the body is at exactly 11 pages. If you want to
raise it, change `MIN_PT` and the rcParams in `analysis/figstyle.py` and expect
to re-tune two or three layouts.

## How to rebuild and re-check

```bash
cd paper/asplos
../tmlr/tectonic.exe -X compile asplos.tex --outdir . --keep-intermediates
python analysis/verify_claims.py      # 49 numbers + anonymity
python analysis/verify_citations.py   # every reference against arXiv/OpenAlex
grep -a "newlabel{concend}" asplos.aux   # page the body ends on
```
