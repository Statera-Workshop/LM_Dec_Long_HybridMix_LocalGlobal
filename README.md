<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-lm_dec_long_hybridmix_localglobal)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# LM_Dec_Long_HybridMix_LocalGlobal

HybridMix_LocalGlobal - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (6 seed-repeat runs, see results)**
- Failure class: Note: best 1B-token quality (1.315) and length-general state tracking (0.92-0.96 at 8192, 5 runs); memory update exact after long training (3/3); fails distant key-value recall (LM-05)

## Provenance

- Source repository: (new) hybrid: GRU + local and global attention
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7 + torch.nn

## Differences from the source ([Unified] protocol)

- GRU layer, 3 sliding-window (256) RoPE layers, then 3 global NoPE layers

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.74 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.688 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.395 / test 1.378 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.321 / test 1.315 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 589 MB, inference 232 MB
- Peak GPU memory at length 8192 (batch 1): train 5436 MB, inference 2951 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.27 (keys removed 0.26); 4 pairs @ 2048: 0.26 (keys removed 0.27); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.06)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.29 (keys removed 0.24); 4 pairs @ 2048: 0.25 (keys removed 0.26); 4 pairs @ 8192: 0.26 (keys removed 0.23); 16 pairs @ 512: 0.07 (keys removed 0.08); 16 pairs @ 2048: 0.07 (keys removed 0.08); 16 pairs @ 8192: 0.07 (keys removed 0.06)
- LM-05 recall `lm05v3s2`: 4 pairs @ 512: 0.25 (keys removed 0.27); 4 pairs @ 2048: 0.27 (keys removed 0.23); 16 pairs @ 512: 0.07 (keys removed 0.05); 16 pairs @ 2048: 0.06 (keys removed 0.06)
- LM-05 recall `lm05v3s3`: 4 pairs @ 512: 0.27 (keys removed 0.26); 4 pairs @ 2048: 0.26 (keys removed 0.25); 16 pairs @ 512: 0.07 (keys removed 0.06); 16 pairs @ 2048: 0.07 (keys removed 0.07)
- LM-05 recall `lm05v3longs2`: 4 pairs @ 512: 0.28 (keys removed 0.30); 4 pairs @ 2048: 0.27 (keys removed 0.26); 4 pairs @ 8192: 0.23 (keys removed 0.28); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.08 (keys removed 0.07); 16 pairs @ 8192: 0.06 (keys removed 0.07)
- LM-06 memory update `lm06`: @512: latest 0.73, unchanged 0.06, stale 0.01; @2048: latest 0.66, unchanged 0.05, stale 0.02
- LM-06 memory update `lm06long`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00
- LM-06 memory update `lm06s2`: @512: latest 0.69, unchanged 0.06, stale 0.01; @2048: latest 0.57, unchanged 0.08, stale 0.03
- LM-06 memory update `lm06s3`: @512: latest 1.00, unchanged 0.99, stale 0.00; @2048: latest 1.00, unchanged 0.99, stale 0.00
- LM-06 memory update `lm06longs2`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hyb_mix
python run.py lm02 hyb_mix      # enwik8 is downloaded to data/ on first use
```
