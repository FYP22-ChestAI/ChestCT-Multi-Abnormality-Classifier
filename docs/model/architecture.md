# Model architecture: stages 2-4

How the model side is built on top of the preprocessing cache, what each stage hands to the next,
and what is still to do. To *run* stage 2, see [README.md](README.md). The input side is described in
[../preprocessing/data_contract.md](../preprocessing/data_contract.md).

```
stage 1  ct_preprocessing   raw CT ─► data/cache/{volume_id}.npy        (N, 224, 224) int16 HU      DONE
stage 2  ct_model.encoders  each slice ─► one feature vector             (N, 1024) per volume       DONE
                            └► data/embeddings/<encoder>-<fp>/{volume_id}.npy   (phase 1: precomputed)
stage 3  ct_model.aggregators  bag of N slice vectors ─► one volume vector (+ attention over slices)   TODO
stage 4  ct_model.heads        volume vector ─► 18 abnormality logits                                  TODO
```

## Roadmap

| Phase | Trains | Reads | Status |
|---|---|---|---|
| **1** | stage 3 (ABMIL) + stage 4, encoder **frozen** | the embedding store (precomputed once) | stage 2 done; stages 3-4 TODO |
| **2** | LoRA adapters in the encoder (+ head) | the HU cache, end to end, k slices per volume | TODO (`encoders/lora.py`) |
| **3** | other aggregators: query-based MIL / QGMIL | the embedding store (or phase-2 encoder) | TODO (`aggregators/query_mil.py`) |

Phase 2 targets the domain gaps: sharp vs soft kernels (soft volumes from a cache subset run) and the
local NHRD scanners. Every TODO module has a docstring with its intended design, inputs and outputs.

| TODO | Where | Notes |
|---|---|---|
| Gated ABMIL | `src/ct_model/aggregators/abmil.py` | mask padding before the softmax; tests listed in the docstring |
| Linear multi-label head | `src/ct_model/heads/linear.py` | prevalence-initialised bias |
| Volume classifier | `src/ct_model/models/volume_classifier.py` | encoder optional (None = precomputed embeddings) |
| Training loop, losses, metrics | `src/ct_model/training/` | outputs to `outputs/experiments/<name>/<timestamp>/` |
| Experiment config + script | `configs/model/experiments/abmil_dale2s.yaml`, `scripts/model/train_mil.py` | schema sketch exists |
| LoRA | `src/ct_model/encoders/lora.py` | peft on `attn.qkv` / `attn.proj`; `lora:` in the encoder YAML |
| Query MIL / QGMIL | `src/ct_model/aggregators/query_mil.py` | one query per label, per-label attention maps |

Already usable by stages 3-4: `EmbeddingBagDataset` + `collate_bags` (padded bags, masks, labels with a
missing-label mask), `SliceSampler`, `select_volumes`, `open_store(..., create=False)`, `seed_everything`.

## Code layout

```
configs/model/
  encode.yaml                 stage-2 defaults: which volumes, batch sizes, device, store folder
  encoders/<name>.yaml        one backbone each: architecture, pinned weights, input transform, pooling
  experiments/<name>.yaml     stage 3+4 experiments (TODO)
src/ct_model/
  config.py                   typed, strict YAML loading; the encoder fingerprint
  data/                       volumes.py (manifest -> records), slices.py (samplers), datasets.py
  encoders/                   base.py (SliceEncoder), transforms.py, timm_vit.py, registry.py, lora.py (TODO)
  embeddings/                 store.py (the embedding store), extract.py (the stage-2 loop)
  aggregators/ heads/ models/ training/     stage 3-4 interfaces and TODO stubs
  utils/                      device.py, seed.py
scripts/model/                encode_volumes.py, check_encoder.py, train_mil.py (TODO)
notebooks/                    encoder_embeddings_report.ipynb (visual checks of stage 2)
tests/model/                  pytest, synthetic data and a tiny random ViT: no download, no GPU
```

The rule from preprocessing carries over: **configs and scripts run pipelines; notebooks only look at
results.** Every number lives in a YAML file; a script argument may override it for one run, and the
settings actually used are printed first.

## Stage 2: the slice encoder

### One interface, many backbones

`SliceEncoder.forward(hu)` takes raw HU slices `(K, H, W)` and returns `(K, D)`. The encoder **owns its
input transform**, because backbones disagree about inputs. DALE-CT wants one z-scored channel;
ImageNet-pretrained models want the data contract's three HU windows as RGB plus ImageNet normalisation.
If windowing lived in the dataset, switching backbone would silently feed it the wrong input.

A backbone is a YAML file. Any timm ViT needs no code (`type: timm_vit`); another family registers a
builder in `encoders/registry.py`. Each backbone is a separate run: `--encoder <name>` writes to its own
store folder, and nothing is ever overwritten.

### DALE-CT-2S (the first backbone)

Taken from the model's `config.json` and model card, and checked by tests:

| | |
|---|---|
| Architecture | timm `vit_large_patch14_dinov2` with `patch_size=16, img_size=512, in_chans=1, num_classes=0, dynamic_img_size=True`; 303.7 M parameters |
| Weights | `Kentucky-Open-Science/DALE-CT-2S`, `model.safetensors`, pinned to commit `a29a028…`; loaded **strictly** (the card's snippet uses `strict=False`, which would hide a mismatched key) |
| Input | 1 channel: `(clip(HU, -997, 888) + 142.39) / 360.97`. This equals the card's "clip → [0, 1] → z-score" exactly (the [0, 1] step cancels); `test_dale_transform_equals_model_card` checks it against the card's code |
| Size | our 224×224 slices, 14×14 patches; position embeddings (32×32 at 512) are interpolated by timm |
| Per-slice feature | CLS token, 1024-d, the same as `model(x)` for `global_pool="token"` |

**Why 224 and not 512:** the cache is 224×224 (fixed in preprocessing, and the raw data is gone). The
slices are already body-cropped, so a 224 slice spans roughly what DALE's 256-px global training crops
covered (60-100 % of a 512 slice). Upsampling to 512 would cost ~5× the compute and add no information.
`input.size_hw` (e.g. 256 or 448, resized on the GPU) is a config value, so this can be tested as an
ablation with a new store.

**Known domain gaps** (motivation for phase 2):
- **Aspect.** Preprocessing stretches the body box to a square (`resize_mode: stretch`), but DALE saw
  undistorted slices.
- **Resolution.** We feed 224 into a 512-native model (see above).
- **Kernel and scanner.** DALE was trained on CT-RATE, but its own card reports frozen-probe AUROC
  dropping on RAD-ChestCT (0.63 frozen vs 0.74 retrained probe), and the NHRD scanners differ again.

### The embedding store (the stage-2 → stage-3 contract)

```
data/embeddings/<encoder name>-<fingerprint8>/      e.g. dale_ct_2s-4f144ab5/
  .embedding_record.json    encoder config + fingerprint, weights commit, HU-cache fingerprint, dim, dtype, versions
  {volume_id}.npy           (n_slices, D) float16; row i = cache slice i (head -> foot)
  index.csv                 volume_id, n_slices, embed_dim
  failed_<source>_<run>.csv only if some volumes failed, with the reason
```

- **The HU cache stays the source of truth.** The store is derived: it can be deleted and rebuilt
  (`encode_volumes.py`), and is never edited by hand.
- **Shared by every run**, like the HU cache. A later run (a soft-kernel subset, NHRD) encodes only the
  volumes the store lacks.
- **One folder per encoder config.** The folder name carries the config fingerprint, so changing
  anything (backbone, weights, input size, pooling, dtype) starts a new folder instead of mixing.
- **Refuses to mix.** If the HU cache's `.cache_fingerprint.json` or the weights commit differ from the
  store's record, the store raises `StoreMismatch`.
- **Row order is slice order.** Stage-3 attention over rows maps straight back to slices, for the
  report notebooks.
- **Missing embeddings are an error.** Stages 3-4 open the store with `open_store(..., create=False)`
  and build `EmbeddingBagDataset`, which raises if any selected volume lacks embeddings. It never
  silently skips a volume.

**float16 is lossless here.** On the GPU the encoder runs under bfloat16 autocast. Every bfloat16
value in fp16's normal range is exactly representable in float16, which has 10 mantissa bits vs 7
(`test_float16_storage_is_lossless_for_bfloat16_outputs`). Values the format cannot hold (NaN, inf,
|x| ≥ 65504) are refused. DALE's outputs peak around |x| ≈ 20. On the CPU (float32, no autocast),
float16 rounds to ~3 significant digits, far below anything a classifier can use.
`store_dtype: float32` exists but doubles the size.

**Why CLS only, not patch tokens:** CLS is the per-slice summary the backbone was trained to produce,
and the standard MIL input. Storing all 196 patch tokens would take ~90 MB per volume (~0.9 TB for
10k volumes). `pooling: mean_patch` or `cls_mean` (2048-d) are available as a new store, if wanted.

## Memory and disk budget (server: 32 GB RAM, one 16 GB RTX 4070-class GPU, ~143 GB free)

| | |
|---|---|
| Store size | ~230 slices × 1024 × 2 B ≈ **0.47 MB per volume**; 10k volumes ≈ **4.7 GB** (≤ 8 GB at 400 slices) |
| GPU, frozen encode | ViT-L in bf16 ≈ 0.6 GB of weights, plus activations for `slice_batch_size` slices of 197 tokens; 128 fits easily. On CUDA OOM the batch halves itself and stays halved |
| Host RAM | each volume is read from the mmap'd cache (~25 MB int16) and stays int16 until it is on the GPU; `num_workers` × prefetch volumes in flight |
| Stage 3-4 training | bags are 0.47 MB each; all 10k volumes fit in RAM (~5 GB as fp16) |
| Phase 2 (LoRA) | backprop through ViT-L for ~230 slices does not fit in 16 GB. Use `SliceSampler("random_k", k=32..64)`, gradient checkpointing, bf16, 1-2 volumes per step + gradient accumulation |
