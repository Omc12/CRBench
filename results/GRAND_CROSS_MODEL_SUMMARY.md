# CRBench Grand Cross-Model Synthesis: E2B vs. E4B vs. 7B

All benchmarks evaluated under canonical **Option B Physical Memory Aggregation** ($S_{\text{res}} = 0.70 \cdot Q_{\text{abs}} + 0.30 \cdot R_{\text{mem}}^{\text{agg}}$) and Canonical Part 2 Hardware Serving Utility ($S_{\text{sys}} = 0.70 \cdot S_{\text{res}} + 0.30 \cdot R_{\text{sys}}$).

## 1. Part 1: Algorithmic Representation Fidelity ($S_{\text{res}}$)

| Method | Gemma 4 E2B (2K-128K) | | | Gemma 4 E4B (2K-32K) | | | Qwen2.5 7B (2K-32K) | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Accuracy** | **$b_{\text{eff}}^{\text{agg}}$** | **$S_{\text{res}}$** | **Accuracy** | **$b_{\text{eff}}^{\text{agg}}$** | **$S_{\text{res}}$** | **Accuracy** | **$b_{\text{eff}}^{\text{agg}}$** | **$S_{\text{res}}$** |
| **`dkv_mid`** | 44.7% | 5.46 bpt | **51.0** | 55.3% | 5.00 bpt | **59.4** | 53.3% | 5.84 bpt | **56.3** |
| **`dkv_high`** | 46.7% | 10.78 bpt | **42.5** | 61.3% | 10.68 bpt | **52.9** | 59.1% | 12.14 bpt | **48.6** |
| **`snapkv`** | 46.8% | 6.00 bpt | **51.5** | 60.0% | 6.00 bpt | **60.8** | 58.2% | 6.00 bpt | **59.5** |
| **`kivi`** | 46.4% | 6.01 bpt | **51.2** | 62.0% | 6.04 bpt | **62.1** | 60.7% | 6.04 bpt | **61.1** |
| **`dense_fp16`** | 46.4% | 16.00 bpt | **32.5** | 64.0% | 16.00 bpt | **44.8** | 60.0% | 16.00 bpt | **42.0** |
| **`low_rank_kv`** | 31.2% | 6.07 bpt | **40.5** | 48.8% | 6.27 bpt | **52.4** | 28.6% | 6.09 bpt | **38.6** |
| **`streaming_llm`** | 22.9% | 6.00 bpt | **34.8** | 25.5% | 6.00 bpt | **36.6** | 36.5% | 6.00 bpt | **44.3** |
| **`kv_merging`** | 13.3% | 6.01 bpt | **28.0** | 25.3% | 6.03 bpt | **36.4** | 19.9% | 6.03 bpt | **32.6** |


## 2. Part 2: Hardware System Serving Utility ($S_{\text{sys}}$)

| Method | Gemma 4 E2B ($S_{\text{sys}}$) | Gemma 4 E4B ($S_{\text{sys}}$) | Qwen2.5 7B ($S_{\text{sys}}$) | Cross-Model Mean $S_{\text{sys}}$ |
| :--- | :---: | :---: | :---: | :---: |
| **`kivi`** | 64.6 | 73.4 | 72.8 | **70.3** |
| **`dkv_mid`** | 59.1 | 66.3 | 54.2 | **59.9** |
| **`low_rank_kv`** | 57.4 | 65.9 | 54.9 | **59.4** |
| **`dense_fp16`** | 52.8 | 61.4 | 59.4 | **57.8** |
| **`streaming_llm`** | 54.3 | 55.6 | 61.0 | **57.0** |
| **`snapkv`** | 60.6 | 48.6 | 53.7 | **54.3** |
| **`dkv_high`** | 52.7 | 60.0 | 48.5 | **53.7** |
| **`kv_merging`** | 49.6 | 55.5 | 52.8 | **52.6** |
