# CRBench Leaderboard: Gemma 4 E4B

### Part 1: Algorithmic Representation Fidelity ($S_{\text{res}} = 0.70 \cdot Q_{\text{abs}} + 0.30 \cdot R_{\text{mem}}^{\text{agg}}$)

| Rank | Method | Accuracy ($Q_{\text{abs}}$) | Realized $b_{\text{eff}}^{\text{agg}}$ | Physical VRAM Saved | Part 1 Score ($S_{\text{res}}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **`kivi_style_kv_quant`** | 62.0% | 6.04 bpt | 62.2% saved | **62.1** |
| **2** | **`snapkv`** | 60.0% | 6.00 bpt | 62.5% saved | **60.8** |
| **3** | **`dkv_mid`** | 55.3% | 5.00 bpt | 68.7% saved | **59.4** |
| **4** | **`dkv_high`** | 61.3% | 10.68 bpt | 33.3% saved | **52.9** |
| **5** | **`low_rank_kv`** | 48.8% | 6.27 bpt | 60.8% saved | **52.4** |
| **6** | **`dense_fp16`** | 64.0% | 16.00 bpt | 0.0% saved | **44.8** |
| **7** | **`streaming_llm`** | 25.5% | 6.00 bpt | 62.5% saved | **36.6** |
| **8** | **`kv_merging`** | 25.3% | 6.03 bpt | 62.3% saved | **36.4** |


### Part 2: Hardware System Serving Utility ($S_{\text{sys}} = 0.70 \cdot S_{\text{res}} + 0.30 \cdot R_{\text{sys}}$)

| Rank | Method | System Score ($S_{\text{sys}}$) | Part 1 Score ($S_{\text{res}}$) | Mean TTFT | Decode Speed | Peak Resident VRAM | Hardware Multiplier ($\phi$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`kivi_style_kv_quant`** | **73.4** | 62.1 | 2.84s | 12.8 tok/s | 761.3 MB | 1.00x |
| **2** | **`dkv_mid`** | **66.3** | 59.4 | 5.06s | 12.3 tok/s | 760.7 MB | 0.83x |
| **3** | **`low_rank_kv`** | **65.9** | 52.4 | 3.18s | 13.1 tok/s | 761.5 MB | 0.97x |
| **4** | **`dense_fp16`** | **61.4** | 44.8 | 2.96s | 12.4 tok/s | 761.5 MB | 1.00x |
| **5** | **`dkv_high`** | **60.0** | 52.9 | 6.25s | 12.2 tok/s | 761.0 MB | 0.76x |
| **6** | **`streaming_llm`** | **55.6** | 36.6 | 2.92s | 13.6 tok/s | 761.4 MB | 1.00x |
| **7** | **`kv_merging`** | **55.5** | 36.4 | 2.91s | 13.5 tok/s | 761.5 MB | 1.00x |
| **8** | **`snapkv`** | **48.6** | 60.8 | 83.38s | 10.6 tok/s | 2632.4 MB | 0.20x |
