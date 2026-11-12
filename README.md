# LM_Dec_Long_HybridMix_LocalGlobal

HybridMix_LocalGlobal - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Smoke-Tested**
- Failure class: Experiment Incomplete: benchmarks not finished

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
- Peak GPU memory at length 2048 (batch 1): train 589 MB, inference 232 MB
- Peak GPU memory at length 8192 (batch 1): train 5436 MB, inference 2951 MB

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
