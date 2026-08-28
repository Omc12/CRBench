# CRBench Leaderboard: Qwen2.5 7B

### Part 1: Algorithmic Representation Fidelity ($S_{\text{res}} = 0.70 \cdot Q_{\text{abs}} + 0.30 \cdot R_{\text{mem}}^{\text{agg}}$)

| Rank | Method | Accuracy ($Q_{\text{abs}}$) | Realized $b_{\text{eff}}^{\text{agg}}$ | Physical VRAM Saved | Part 1 Score ($S_{\text{res}}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **`kv_quant`** | 60.7% | 6.04 bpt | 62.2% saved | **61.1** |
| **2** | **`snapkv`** | 58.2% | 6.00 bpt | 62.5% saved | **59.5** |
| **3** | **`dkv_mid`** | 53.3% | 5.84 bpt | 63.5% saved | **56.3** |
| **4** | **`dkv_high`** | 59.1% | 12.14 bpt | 24.1% saved | **48.6** |
| **5** | **`streaming_llm`** | 36.5% | 6.00 bpt | 62.5% saved | **44.3** |
| **6** | **`dense_fp16`** | 60.0% | 16.00 bpt | 0.0% saved | **42.0** |
| **7** | **`low_rank_kv`** | 28.6% | 6.09 bpt | 62.0% saved | **38.6** |
| **8** | **`kv_merging`** | 19.9% | 6.03 bpt | 62.3% saved | **32.6** |


### Part 2: Hardware System Serving Utility ($S_{\text{sys}} = 0.70 \cdot S_{\text{res}} + 0.30 \cdot R_{\text{sys}}$)

| Rank | Method | System Score ($S_{\text{sys}}$) | Part 1 Score ($S_{\text{res}}$) | Mean TTFT | Decode Speed | Peak Resident VRAM | Hardware Multiplier ($\phi$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`kv_quant`** | **72.8** | 61.1 | 5.46s | 16.4 tok/s | 3854.3 MB | 1.00x |
| **2** | **`streaming_llm`** | **61.0** | 44.3 | 5.51s | 24.9 tok/s | 3854.4 MB | 1.00x |
| **3** | **`dense_fp16`** | **59.4** | 42.0 | 5.56s | 16.3 tok/s | 3854.2 MB | 1.00x |
| **4** | **`low_rank_kv`** | **54.9** | 38.6 | 6.82s | 18.1 tok/s | 3854.4 MB | 0.93x |
| **5** | **`dkv_mid`** | **54.2** | 56.3 | 39.55s | 15.3 tok/s | 3854.2 MB | 0.49x |
| **6** | **`snapkv`** | **53.7** | 59.5 | 58.69s | 22.2 tok/s | 5132.6 MB | 0.40x |
| **7** | **`kv_merging`** | **52.8** | 32.6 | 5.51s | 25.8 tok/s | 3854.4 MB | 1.00x |
| **8** | **`dkv_high`** | **48.5** | 48.6 | 41.88s | 15.3 tok/s | 3854.5 MB | 0.48x |
