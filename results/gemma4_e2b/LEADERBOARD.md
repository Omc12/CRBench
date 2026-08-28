# CRBench Leaderboard: Gemma 4 E2B

### Part 1: Algorithmic Representation Fidelity ($S_{\text{res}} = 0.70 \cdot Q_{\text{abs}} + 0.30 \cdot R_{\text{mem}}^{\text{agg}}$)

| Rank | Method | Accuracy ($Q_{\text{abs}}$) | Realized $b_{\text{eff}}^{\text{agg}}$ | Physical VRAM Saved | Part 1 Score ($S_{\text{res}}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **`snapkv`** | 46.8% | 6.00 bpt | 62.5% saved | **51.5** |
| **2** | **`kivi_style_kv_quant`** | 46.4% | 6.01 bpt | 62.4% saved | **51.2** |
| **3** | **`dkv_mid`** | 44.7% | 5.46 bpt | 65.9% saved | **51.0** |
| **4** | **`dkv_high`** | 46.7% | 10.78 bpt | 32.6% saved | **42.5** |
| **5** | **`low_rank_kv`** | 31.2% | 6.07 bpt | 62.1% saved | **40.5** |
| **6** | **`streaming_llm`** | 22.9% | 6.00 bpt | 62.5% saved | **34.8** |
| **7** | **`dense_fp16`** | 46.4% | 16.00 bpt | 0.0% saved | **32.5** |
| **8** | **`kv_merging`** | 13.3% | 6.01 bpt | 62.5% saved | **28.0** |


### Part 2: Hardware System Serving Utility ($S_{\text{sys}} = 0.70 \cdot S_{\text{res}} + 0.30 \cdot R_{\text{sys}}$)

| Rank | Method | System Score ($S_{\text{sys}}$) | Part 1 Score ($S_{\text{res}}$) | Mean TTFT | Decode Speed | Peak Resident VRAM | Hardware Multiplier ($\phi$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`kivi_style_kv_quant`** | **64.6** | 51.2 | 12.51s | 11.9 tok/s | 847.0 MB | 0.96x |
| **2** | **`snapkv`** | **60.6** | 51.5 | 13.04s | 12.9 tok/s | 2088.2 MB | 0.82x |
| **3** | **`dkv_mid`** | **59.1** | 51.0 | 22.26s | 13.8 tok/s | 1389.9 MB | 0.78x |
| **4** | **`low_rank_kv`** | **57.4** | 40.5 | 12.80s | 12.3 tok/s | 849.8 MB | 0.97x |
| **5** | **`streaming_llm`** | **54.3** | 34.8 | 12.58s | 13.7 tok/s | 847.4 MB | 1.00x |
| **6** | **`dense_fp16`** | **52.8** | 32.5 | 13.18s | 13.4 tok/s | 1118.4 MB | 1.00x |
| **7** | **`dkv_high`** | **52.7** | 42.5 | 23.40s | 13.8 tok/s | 1391.0 MB | 0.77x |
| **8** | **`kv_merging`** | **49.6** | 28.0 | 12.59s | 13.7 tok/s | 847.8 MB | 1.00x |
