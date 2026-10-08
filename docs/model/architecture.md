# Model architecture: stages 2-4

How the model side is built on top of the preprocessing cache, what each stage hands to the next, why
it is built that way, and what is still to do. To *run* anything, see [README.md](README.md). The input
side is described in [../preprocessing/data_contract.md](../preprocessing/data_contract.md).

```
stage 1  ct_preprocessing      raw CT ─► data/cache/{volume_id}.npy         (N, 224, 224) int16 HU     DONE
stage 2  ct_model.encoders     each slice ─► one feature vector              (N, 1024) per volume      DONE
                               └► data/embeddings/<encoder>-<fp>/{volume_id}.npy   (phase 1: precomputed)
stage 3  ct_model.aggregators  N slice vectors ─► one volume vector (+ attention over slices)          DONE: ABMIL, mean pool
stage 4  ct_model.heads        volume vector ─► 18 abnormality logits                                  DONE: linear head
         ct_model.explain      optional evidence: which slices (exact), where in a slice (Grad-CAM)    DONE
```

## Roadmap

| Phase | Trains | Reads | Status |
|---|---|---|---|
| **1** | stage 3 (ABMIL) + stage 4, encoder **frozen** | the embedding store (precomputed once) | **built**; to run on the server (`train-sharp-5800`) |
| **2** | LoRA adapters in the encoder (+ head) | the HU cache, end to end, k slices per volume | TODO (`encoders/lora.py`) |
| **3** | other aggregators: query-based MIL / QGMIL | the embedding store (or phase-2 encoder) | TODO (`aggregators/query_mil.py`) |

Phase 2 targets the domain gaps: sharp vs soft kernels (soft volumes from a cache subset run) and the
local NHRD scanners. Every TODO module has a docstring with its intended design, inputs and outputs.

| TODO | Where | Notes |
|---|---|---|
| LoRA | `src/ct_model/encoders/lora.py` | peft on `attn.qkv` / `attn.proj`; `lora:` in the encoder YAML; train from the HU cache with `SliceSampler("random_k")` |
| Query MIL / QGMIL | `src/ct_model/aggregators/query_mil.py` | registered name already; one query per label, `per_label = True` |
| Original-scan slice numbers in evidence | `ct_preprocessing` (crop box) | `crop.py` computes the crop box but the sidecar does not store it, so evidence names slices of the *preprocessed* volume |

## Code layout

```
configs/model/
  encode.yaml                 stage-2 defaults: which volumes, batch sizes, device, store folder
  encoders/<name>.yaml        one backbone each: architecture, pinned weights, input transform, pooling
  experiments/<name>.yaml     stages 3+4: abmil_dale2s, abmil_perlabel_dale2s, meanpool_dale2s
src/ct_model/
  config.py                   typed, strict YAML loading; the encoder fingerprint
  registry.py                 type name -> builder, for encoders, aggregators, heads, losses
  data/                       volumes.py (manifest -> records), slices.py (samplers), datasets.py
  encoders/                   base.py (SliceEncoder), transforms.py, timm_vit.py, registry.py, lora.py (TODO)
  embeddings/                 store.py (the embedding store), extract.py (the stage-2 loop)
  aggregators/                base.py (interface), abmil.py, mean_pool.py, query_mil.py (TODO)
  heads/                      base.py, linear.py
  models/                     volume_classifier.py: FeatureNorm + aggregator + head, exact slice contributions
  training/                   config.py, data.py, losses.py, metrics.py, trainer.py, evaluate.py, records.py
  explain/                    evidence.py (L1), heatmap.py (L2), volume.py (one volume, plots)
  utils/                      device.py, seed.py
scripts/model/                encode_volumes.py, check_encoder.py, train_mil.py, evaluate_mil.py,
                              summarize_experiments.py, explain_volume.py
notebooks/                    encoder_embeddings_report, experiments_report, evidence_viewer
tests/model/                  pytest, synthetic data and a tiny random ViT: no download, no GPU
outputs/experiments/          (git-ignored) every training run, its evaluations, and the two ledgers
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

## Stages 3-4: aggregation and classification (phase 1)

### What aggregation does

A volume is a *bag* of ~200-250 slice embeddings (1024-d each) with one label vector for the whole
volume, which is multiple-instance learning (MIL). Stage 3 combines all of a volume's slice embeddings
into **one volume embedding**, using a *learned weighted average*:
- each slice gets a weight `a_k ≥ 0`, and the weights sum to 1;
- it is permutation-invariant, so slice order does not change the result;
- it works for any slice count, because padding is masked out.

Stage 4 maps that embedding to 18 logits. Unlike a plain average, the weights let the model emphasise
the few slices that show a finding.

### ABMIL, as implemented

Gated attention MIL from Ilse, Tomczak & Welling, ICML 2018 (arXiv 1802.04712). The equations were read
from the paper and are checked by `tests/model/test_mil_components.py`:

```
h_k = Dropout(ReLU(Linear_{1024→256}(x̂_k)))                    x̂ = standardised slice embedding
a_k = softmax_k( wᵀ( tanh(V h_k) ⊙ sigm(U h_k) ) )              Eq. 9; V, U: 256→128; padded slices get exactly 0
z   = Σ_k a_k h_k                                               Eq. 7
```

V and U are `nn.Linear` layers, so they carry biases that the paper's equation omits; the test sets
them to zero and checks Eq. 9 exactly.

`branches` decides how the attention is shared:
- `single` is the paper: one attention and one `z`, shared by all 18 labels.
- `per_label` gives each label its own `w`, so each label has its own attention over slices and its own
  `z_c`.

**Why `per_label` is a first-class option.** On a synthetic test set where each finding lives in its own
slices, both variants were run with 2 seeds each:
- single-branch ABMIL reached val macro AUROC ≈ 0.82, and the rarest label stalled (0.66);
- `per_label` reached ≈ 0.94-0.97;
- an oracle reached ≈ 0.97-1.0.

One shared attention cannot cover several findings in different slices at once. That is a measurement
on synthetic data, *not* a result on CT-RATE: the real comparison is `abmil_dale2s` vs
`abmil_perlabel_dale2s` on val. `per_label` also gives label-specific slice evidence.

**Baseline.** `mean_pool` uses the same projection with uniform weights. Every attention model must
beat it on val to justify itself.

### Head, prior, standardisation

- **Head:** `LinearHead` computes `logit_c = w_c · z + b_c`, or `w_c · z_c + b_c` with per-label pooling.
- **Prior:** `b_c` is initialised to `log(p_c / (1 - p_c))` from **train** prevalence, so the untrained
  model predicts each label's base rate. Tested: `σ(b_c) = p_c`.
- **Standardisation:** each embedding dimension is standardised with a mean / std fitted on **train**
  slices only. These are stored as buffers in the checkpoint, so inference needs nothing else.

### Class imbalance

**The data (`train-sharp-5800` train split).**
- Prevalence ranges from 46.6 % (lung nodule) down to 7.4 % (pericardial effusion), so negatives per
  positive run from 1.1 to 12.6.
- Val and test prevalence are within 3 points of train.
- The imbalance is moderate; it is not extreme.

**Always applied:**
- the prevalence-initialised head bias;
- per-label decision thresholds chosen on **val** (maximum F1).

AUROC and AUPRC are threshold-free and do not need thresholds. Sensitivity, specificity and F1 for rare
labels depend almost entirely on the threshold, so the thresholds are what fix them.

**Compared by experiment, not assumed:**
- `bce`;
- `weighted_bce`: `pos_weight` = train negatives / positives, i.e. `torch.nn.BCEWithLogitsLoss` with `pos_weight`;
- `asl`: Asymmetric Loss, Ridnik et al., ICCV 2021, as implemented in the authors' reference code
  `Alibaba-MIIL/ASL`, with its defaults `gamma_neg=4, gamma_pos=1, clip=0.05` and a constant focusing
  weight.

Every loss is a mean over unmasked entries, so the learning rate means the same for all three. ASL is
checked against an independent re-implementation of the reference code, including its gradient. The
choice is made on val macro **AUPRC** (more sensitive to rare labels than AUROC) and macro AUROC,
averaged over seeds.

**Not used: resampling.** In multi-label data, oversampling a volume for a rare label also oversamples
every label it co-occurs with (3.4 findings per volume on average). That distorts the others in ways a
loss weight does not.

### Training schedule: how many iterations

`ceil(5,827 train volumes / 16 per batch) = 365` steps per epoch.

**Budget.** At most **100 epochs = 36,500 steps**: AdamW, linear warm-up for 2 epochs, then cosine to 0;
gradient clipping 1.0; weight decay on weight matrices only.

**Stopping rule.** Training stops after **15 epochs without a better val macro AUROC**, and the best
epoch's checkpoint is kept. The best epoch is **measured and recorded** (`run_info.json`, `index.csv`),
not predicted. If it lands in the last 10 % of the budget, the report notebook flags it: raise
`optim.max_epochs` for that config.

**Cost.** Bags are preloaded into RAM (~3.3 GB for train + val), so an epoch is seconds on the GPU and
5-seed comparisons are affordable.

### Evaluation protocol

- **Choices on val, test once.** Every choice (loss, branches, learning rate, thresholds) is made on
  **val**. `train_mil.py` never reads the test split; a test checks this.
- **Evaluation.** `evaluate_mil.py` scores a finished run on test **once**:
  - It uses the best checkpoint and the val thresholds, unchanged.
  - Results are reported overall, **per kernel class**, and **per scanner manufacturer**. Training saw
    only sharp, so sharp vs soft is the cross-kernel result; 10 of the 18 labels depend on the scanner.
  - A group under `--min-group` volumes (default 100; e.g. the 36 `other`-kernel test volumes) is
    flagged `too_small` and gets no CI.
- **Uncertainty.** 95 % bootstrap CIs resample **patients**, because the sharp and soft reconstructions
  of one scan are not independent.
- **NaN labels.** A label with a single class in a group is NaN, never dropped silently.

### Experiment identity and records

The **config hash** identifies what was learned. It hashes:
- the resolved experiment config, without name, seed, device and paths;
- the embedding store folder name (encoder, weights commit and preprocessing, through its fingerprint);
- the sha256 of the run manifest (volumes, splits, labels).

Same hash means same experiment, so seeds sit side by side and average. Any change gives a new hash.

```
outputs/experiments/
  index.csv          one row per finished training run (ledger)
  evaluations.csv    one row per evaluation (ledger)
  <name>/<hash8>/seed<k>/
      config.yaml  run_info.json (git commit + dirty flag, versions, GPU, data, best epoch)  labels.json
      store_record.json  checkpoints/{best,last}.pt  metrics.csv  predictions_val.csv  thresholds.json
      val_metrics.json  val_per_label.csv
      eval/<source>__<run>__<split>/   predictions.csv  metrics_summary.csv  metrics_per_label.csv  eval_info.json
      explain/<volume_id>/             evidence.json, one PNG per label
```

Nothing is overwritten:
- a finished run or an existing evaluation is refused;
- an interrupted run resumes from `last.pt` when the same command is run again (a test interrupts
  after epoch 3 and checks the resumed run).

### Evidence (optional, on demand)

**L1: which slices.** With linear pooling and the linear head, a logit is *exactly* a sum of per-slice
terms:

```
logit_c = b_c + Σ_k a_{c,k} (w_c · h_k)
```

`a_{c,k} = a_k` for single-branch ABMIL and `1/N` for mean pool. `slice_evidence` checks the sum on
every call. On synthetic data with planted findings, the top slice is a planted one for ≥ 90 % of
positive volumes (tested).

**L2: where in the slice.** Grad-CAM (Selvaraju et al., ICCV 2017) on the ViT's last-block patch tokens:
1. Re-encode the slice with the stage-2 encoder. It is rebuilt from the embedding store's own record:
   the exact config and weights commit.
2. Put it back into the stored bag and back-propagate the label's logit.
3. Weight the tokens by the mean gradient, apply ReLU, reshape to the 14×14 patch grid, and upsample.

Guards and checks:
- **Re-encode guard.** The re-encoded embedding must match the stored one (cosine ≥ 0.99), otherwise
  the map would explain a different model. A store made by a randomly initialised encoder is refused
  outright.
- **Diagnostics to run on the real model:**
  - randomisation (Adebayo et al., NeurIPS 2018): the map must change when the head is randomised;
  - deletion: blanking the top patches should lower the logit more than blanking random ones.
- **Verified so far:** the Grad-CAM arithmetic matches an independent autograd computation. Whether
  DALE's maps pass the diagnostics is to be checked on the server; it is not established here.

**Limits.** Evidence shows what the *model* relied on; it is not a validated localisation of a finding.
Slice numbers refer to the **preprocessed** volume (1.5 mm, head to foot, body-cropped). The crop box is
not stored yet, so they cannot be mapped back to original DICOM slice numbers.
