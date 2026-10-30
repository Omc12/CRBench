# TMLR resubmission — description of changes

Paste the section below into the "description of changes" field of the new
OpenReview submission, and add the link to the previous submission where
indicated. TMLR's editorial policy requires a resubmission to be entered as a
**new** submission carrying a link to the rejected one plus this description.

Before submitting, check:

- [ ] PDF is `crbench_tmlr_v2.pdf`, built from the unmodified TMLR style file
      (no margin, `\textheight` or `\maketitle` overrides — the previous
      submission carried several).
- [ ] Supplementary ZIP is `crbench_supplement_anon.zip` (15.7 MB, under the
      100 MB cap). Rebuild with `python paper/tmlr/make_supplement.py`; it
      refuses to produce an archive if any identifying string survives.
- [ ] Link to the previous submission filled in below.
- [ ] OpenReview profiles complete for all authors.

---

## Description of changes

This manuscript is a substantially revised version of our previously submitted
manuscript, "CRBench: A Method-Agnostic, Resource-Aware Evaluation Framework for
Long-Context Large Language Models" (TMLR submission **[NUMBER]**, link below).
The underlying experiments are unchanged: the same 3,150 paired query executions
across three models, seven compression methods and five long-context tasks. What
has changed is the scientific framing, the correctness of the reported
quantities, and the disclosure of how the measurements were produced.

**1. The paper is now organised around a finding rather than around an artifact.**
The previous version presented CRBench as an evaluation framework and reported
results as a leaderboard. The revision leads with the empirical claim those
results support: representation efficiency, task utility and end-to-end serving
efficiency are three distinct axes, and none predicts the others. We show that
at an identical nominal 4-bit budget task accuracy spans 2.5%–47.3% on one model
(an 18.7x range that a compression ratio cannot distinguish); that two methods
matched to within 0.55 effective bits per element and 2.0 accuracy points differ
by 30x in time-to-first-token and 3.5x in peak VRAM; and that across 36
operating points the effective footprint of a representation is statistically
uninformative about its realised hardware efficiency. CRBench is introduced as
the instrument these findings require. The title has been changed accordingly to
"Beyond Compression Ratio: Evaluating the Algorithmic and Systems Utility of
Long-Context KV-Cache Compression".

**2. We corrected an error in the reported memory accounting.** Our definition of
effective bits per element has always included representation metadata, but the
aggregate reported in the previous submission counted the compressed payload
only. Recomputing it correctly from the already-collected records raises DKV-Mid
from 5.00 to 6.51 bits per element, DKV-High from 10.68 to 13.07, and a nominal
4-bit quantiser from 4.05 to 4.55, and it reorders the Part 1 table. The
correction penalises the authors' own prior method most; we report that
explicitly. All Part 1 scores in the revision are computed on the
metadata-inclusive figure, with the payload-only figure shown alongside so the
two accountings can be compared directly.

**3. We now report each operating point separately instead of pooling them.**
The previous submission reported one row per method at roughly 6 bits per
element. That row was in fact the average of two separate runs, at nominal 4 and
8 bits, whose accuracies differ by up to 30 points. The revision reports both
budgets, which is what makes the matched-budget comparison — the paper's central
evidence — visible at all.

**4. The Part 2 hardware multiplier is now computed as the paper describes it.**
The previous text stated that the efficiency components were computed per query
and then aggregated; the code computed them from run-level means. They are now
genuinely computed per query against that query's own paired dense execution and
then averaged, consistent with the query-level pairing the framework is built
on.

**5. We corrected the reported experimental environment.** The previous
submission stated NVIDIA RTX GPUs with 16 GB VRAM, CUDA 12.4 and PyTorch 2.4.
The runs were in fact performed on a single RTX 4070 SUPER (12 GB) with PyTorch
2.6.0 (CUDA 12.4 build), transformers 5.15.1 and Python 3.12.10 on Windows 11.

**6. We replaced an inaccurate provenance claim with a per-method disclosure.**
The previous submission stated that all candidate methods were driven from
vendored upstream implementations. That is true for SnapKV, StreamingLLM and
DKV, but KIVI, KV-Merging, Low-Rank KV and the dense reference are CRBench
reference implementations. The revision includes a provenance table stating, for
each method, which implementation produced the numbers, whether its memory
saving was physically realised or analytically accounted, and what upstream
entry point was used. We now state explicitly that SnapKV was driven outside its
fused attention path (its monkeypatch targets transformers 4.37 and
Llama/Mistral, while our runs use transformers 5.x on Gemma and Qwen), so its
measured prefill cost characterises that integration rather than SnapKV as
published; and that our DKV numbers characterise its block compressor rather
than its paged serving runtime. This distinction is now part of the paper's
argument: serving efficiency is a property of an algorithm *and* its
implementation, which is precisely why it cannot be read off a compression
ratio.

**7. We scoped the conclusions and expanded the limitations.** Claims are now
stated as findings about the models, implementations, tasks and hardware
evaluated, not as properties of method families in general. The limitations
section states that Axis 3 measures code paths rather than algorithms; that two
of the five tasks (variable tracking, which every method *and* the dense
reference fails, and multi-needle retrieval, which is near a scoring floor for
the dense reference) carry little discriminative signal; that rank correlations
are computed over 12–13 operating points per model and are descriptive rather
than inferential; and that the single-GPU, batch-size-one setting does not
transfer to multi-tenant serving. We also note that we never correlate S_res
with S_sys, because S_sys contains S_res by construction.

**8. Presentation.** The figures were rebuilt from the corrected data with a
consistent, colour-blind-safe design; a new Figure 1 states the paper's central
claim on the first two pages. The Pareto frontier is now computed within each
model rather than across models, since a cross-model frontier conflates base
model capability with representation quality. Tables and figures are generated
by released scripts, and a verification script re-derives every number quoted in
the prose and fails if the manuscript has drifted from the data.

**9. Anonymisation.** The supplementary archive accompanying the previous
submission contained the authors' GitHub handle in the upstream-repository URLs
recorded in the results manifests, in the adapter bridge, in the reproducibility
guide, and a given name in the LICENSE. The supplement is now produced by a
script that rewrites those strings and refuses to build an archive if any
identifying string survives.

The experimental results themselves are unchanged. Where reported numbers
differ from the previous submission, the difference comes from recomputing the
already-collected per-query records under the accounting the paper defines, not
from new runs.

Previous submission: TMLR submission **[NUMBER]** — **[OpenReview link]**
