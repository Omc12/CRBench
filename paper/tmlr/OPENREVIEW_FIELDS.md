# OpenReview submission fields — copy/paste

Character limits are 5000 for both Abstract and Changes Since Last Submission.
Lengths below are checked by `check_field_lengths.py`.

---

## Title

```
Beyond Compression Ratio: Evaluating the Algorithmic and Systems Utility of Long-Context KV-Cache Compression
```

---

## Abstract

```
Key-value (KV) cache compression is almost always reported as a compression ratio paired with a task score. We show that this pairing is not sufficient to predict how a method will behave in a serving system, and that the two numbers do not even predict each other. Across 3,150 paired query executions spanning seven compression methods, three open-weight models (Gemma 4 E2B, Gemma 4 E4B, Qwen2.5 7B), five long-context tasks and context lengths from 2K to 128K tokens, we find that (i) at an identical nominal budget of 4 bits per KV element, task accuracy ranges from 2.5% to 47.3% on Gemma 4 E2B, an 18.7x spread across methods that a compression ratio cannot distinguish; (ii) the memory a method saves carries essentially no information about the hardware efficiency it realises (Spearman rho = +0.04, p = 0.84 over 36 operating points), and the compressed footprint carries none about prefill cost (rho = +0.01, p = 0.98); and (iii) two methods matched to within 0.55 bits per element and 2.0 accuracy points differ by 30x in time-to-first-token and 3.5x in peak VRAM on the same model and prompts. Representation efficiency, task utility and serving efficiency are three separate axes, and a benchmark that collapses them hides exactly the information a practitioner needs.

We introduce CRBench, the measurement instrument these findings require: a query-level primitive that pairs every compressed execution with an uncompressed dense execution on the identical prompt, an effective-bits-per-element accounting that charges each representation for its mandatory metadata, and a decoupled two-track report that separates algorithmic representation utility from measured serving utility rather than merging them into one score. Charging for metadata is not a formality: it raises a factorised representation from a nominal 5.00 to a realised 6.51 bits per element and a nominal 4-bit quantiser from 4.05 to 4.55, understatements of 30% and 12% that reorder the resulting table. We report per-method provenance, state which memory savings are physically realised and which are analytically accounted, and scope every conclusion to the models, implementations and hardware evaluated. Code, raw per-query records and the scripts that regenerate every number and figure are released.
```

---

## PDF

`paper/tmlr/crbench_tmlr_v2.pdf`

## Beyond PDF

Leave empty.

## Submission Type

**Long submission (more than 12 pages of main content)** — references begin on
page 14, so main content is 13+ pages. See the note at the bottom about
trimming to qualify as Regular.

## Supplementary Material

`paper/tmlr/crbench_supplement_anon.zip` (15.7 MB, anonymised, under the 100 MB cap).

## Previous TMLR Submission Url

```
https://openreview.net/forum?id=LzZFT1mn0z
```

## Changes Since Last Submission

```
This is a substantially revised version of TMLR submission "CRBench: A Method-Agnostic, Resource-Aware Evaluation Framework for Long-Context Large Language Models" (https://openreview.net/forum?id=LzZFT1mn0z), which was desk rejected. The experiments are unchanged: the same 3,150 paired query executions across three models, seven compression methods and five long-context tasks. What changed is the framing, the correctness of the reported quantities, and the disclosure of how they were produced.

1. Reframed around a finding rather than an artifact. The previous version presented CRBench as a framework and reported a leaderboard. This version leads with the claim the results support: representation efficiency, task utility and serving efficiency are three distinct axes, and none predicts the others. The abstract states the supporting measurements. CRBench is introduced as the instrument these findings require, and the title changed accordingly.

2. Corrected an error in the memory accounting. Our definition of effective bits per element has always included representation metadata, but the aggregate reported previously counted the compressed payload only. Recomputing it from the already-collected records raises DKV-Mid from 5.00 to 6.51 bits per element, DKV-High from 10.68 to 13.07, and a nominal 4-bit quantiser from 4.05 to 4.55, and it reorders the Part 1 table. The correction penalises the authors' own prior method most; we report that explicitly.

3. Report each operating point separately instead of pooling. The previous submission reported one row per method at roughly 6 bits per element. That row was in fact the average of two runs, at nominal 4 and 8 bits, whose accuracies differ by up to 30 points. Both are now reported, which is what makes the matched-budget comparison visible at all.

4. The Part 2 hardware multiplier is now computed as described. The previous text said the efficiency components were computed per query and then aggregated; the code computed them from run-level means. They are now genuinely computed per query against that query's own paired dense execution.

5. Corrected the reported environment. The previous submission stated 16 GB VRAM and PyTorch 2.4. Runs were on a single RTX 4070 SUPER (12 GB) with PyTorch 2.6.0 (CUDA 12.4), transformers 5.15.1 and Python 3.12.10 on Windows 11.

6. Replaced an inaccurate provenance claim with a per-method disclosure. The previous submission stated that all methods were driven from vendored upstream implementations. That holds for SnapKV, StreamingLLM and DKV, but KIVI, KV-Merging, Low-Rank KV and the dense reference are our own reference implementations. A provenance table now states, per method, which implementation produced the numbers and whether its memory saving was physically realised or analytically accounted. We state explicitly that SnapKV was driven outside its fused attention path (its monkeypatch targets transformers 4.37 and Llama/Mistral, while we run transformers 5.x), so its measured prefill cost characterises that integration rather than SnapKV as published; and that our DKV numbers characterise its block compressor, not its paged serving runtime. This is now part of the argument: serving efficiency is a property of an algorithm and its implementation, and so cannot be read off a compression ratio.

7. Scoped the conclusions and expanded the limitations. Claims are stated as findings about the models, implementations, tasks and hardware evaluated. The limitations section notes that Axis 3 measures code paths rather than algorithms; that two of five tasks carry little discriminative signal (variable tracking, which every method and the dense reference fails, and multi-needle retrieval, near a scoring floor for dense); that rank correlations over 12-13 operating points are descriptive, not inferential; and that the single-GPU, batch-size-one setting does not transfer to multi-tenant serving. We never correlate S_res with S_sys, since S_sys contains S_res by construction.

8. Presentation and reproducibility. Figures were rebuilt from the corrected data, with a new Figure 1 stating the central claim on the first two pages, and the Pareto frontier is now computed within each model rather than across models. All tables and figures are generated by released scripts, and a verification script re-derives every number quoted in the prose and fails if the manuscript has drifted from the data.

9. Anonymisation. The previous supplementary archive contained the authors' GitHub handle in the results manifests, the adapter bridge and the reproducibility guide, and a given name in the LICENSE. The supplement is now built by a script that refuses to produce an archive if any identifying string survives.

Where numbers differ from the previous submission, the difference comes from recomputing the already-collected per-query records under the accounting the paper defines, not from new runs.
```

## Competing Interests

```
N/A
```

## Human Subjects Reporting

```
N/A
```

## License

CC BY 4.0 (the only option).

## Readers / Signatures

Leave at the defaults.
