# Model pipeline: stages 2-4

- **Stage 2** (this page, first half) turns every volume of a run into per-slice embeddings. Each
  volume's `(N, 224, 224)` int16 HU in `data/cache/` becomes an `(N, 1024)` float16 array in
  `data/embeddings/<encoder>-<fp>/`.
- **Stages 3-4** ([second half](#stages-3-4-train-evaluate-compare)) train ABMIL plus an 18-label head on
  those embeddings, evaluate on test, and keep every result identifiable for later comparisons.
- **[Evidence](#evidence-optional)** is optional: per predicted label, which slices and where in them.

How the stages fit together, and why they are built this way, is in [architecture.md](architecture.md).

The HU cache is the source of truth and is only read. The embedding store is derived: delete it and
re-run to rebuild it.

## Setup (once, on the server)

```bash
python -m pip install -e ".[dev,model]"
```
Installs torch, timm, safetensors and scikit-learn with the package. Use a CUDA build of torch on the server (see
pytorch.org for the right `--index-url`); `python -c "import torch; print(torch.cuda.is_available())"`
must print `True`.

The DALE-CT-2S weights (~1.2 GB, CC BY-NC-SA 4.0) download on first use into the Hugging Face cache
(`~/.cache/huggingface`, or `HF_HOME`). They are public, so no login is needed. The commit is pinned in
`configs/model/encoders/dale_ct_2s.yaml`.

## Run it

Run everything from the repository root. Defaults come from `configs/model/encode.yaml`; every argument
below is optional and overrides it.

**1. Smoke test (a minute; writes nothing).** It loads the backbone and encodes 64 slices of one
volume. It prints the input range after the backbone's HU transform, the output shape, finiteness,
speed and peak GPU memory, plus the projected time for the whole run:
```bash
python scripts/model/check_encoder.py --run train-sharp
```
Raise or lower `--slice-batch-size` and pick the fastest value that fits. Put it in `encode.yaml`.

**2. Dry run.** Shows how many volumes are selected, how many are already encoded, and the store size:
```bash
python scripts/model/encode_volumes.py --run train-sharp --dry-run
```

**3. Encode, inside tmux** so it survives logging out:
```bash
tmux new -s encode
```
```bash
python scripts/model/encode_volumes.py --run train-sharp 2>&1 | tee -a encode_train-sharp.log
```
Detach with `Ctrl+b d`, log out, and come back later; `tmux a -t encode` reattaches (`tmux ls` lists
sessions). The run is resumable: if it stops for any reason (Ctrl-C, the session or the server dies),
run the same command again. Volumes already in the store are skipped, and every file is written
atomically, so nothing is ever half-written. Run one encode at a time per GPU.

The first line of output is the settings actually used. Progress prints every 25 volumes. At the end it
prints counts, slices/s, peak VRAM and the store size. Exit code 1 means some volumes failed (listed with
reasons in `failed_<source>_<run>.csv` in the store folder) or the run stopped on low disk.

**4. Look at it.** Open `notebooks/encoder_embeddings_report.ipynb`, set `RUN`, and run all cells.

### Common variations

```bash
python scripts/model/encode_volumes.py --run soft-subset --kernel-classes soft   # a later soft-kernel run: only missing volumes are encoded
python scripts/model/encode_volumes.py --run train-sharp --splits test            # only the test split
python scripts/model/encode_volumes.py --run train-sharp --encoder dinov2_vitl14   # another backbone -> its own store folder
python scripts/model/encode_volumes.py --run train-sharp --limit 20               # a quick partial run
```

## Script arguments

Every argument is optional. Left out, the value comes from the config file in the "Default" column
(`encode.yaml` = `configs/model/encode.yaml`). The settings actually used are printed as the first line.

### `scripts/model/encode_volumes.py`

| Argument | Values | Default | What it does |
|---|---|---|---|
| `--config PATH` | a YAML file | `configs/model/encode.yaml` | the stage-2 settings file all other defaults come from |
| `--encoder NAME` | a file name in `configs/model/encoders/` (without `.yaml`), or a path to a YAML | `encoder:` in encode.yaml (`dale_ct_2s`) | which backbone; each encoder config writes to its own store folder |
| `--source NAME` | a source in `configs/preprocessing.yaml` (`ctrate`, `nhrd_local`) | `selection.source` (`ctrate`) | whose runs to read |
| `--run NAME` | a run folder in `data/runs/<source>/` | `selection.run` (null = the only run on disk) | which run's manifest selects the volumes |
| `--splits S [S ...]` | any of `train` `val` `test` | `selection.splits` (all three) | only volumes of these splits |
| `--kernel-classes K [K ...]` | e.g. `sharp` `soft` `other` | `selection.kernel_classes` (null = all) | only volumes of these kernel classes |
| `--device D` | `auto` `cuda` `cpu` | `encode.device` (`auto`) | `auto` = the GPU if present; `cuda` without a GPU is an error, not a silent CPU fallback |
| `--amp-dtype T` | `auto` `bfloat16` `float16` `float32` | `encode.amp_dtype` (`auto`) | autocast precision; `auto` = bfloat16 on a GPU that supports it, else float32 |
| `--slice-batch-size N` | integer ≥ 1 | `encode.slice_batch_size` (128) | slices per forward pass; halves itself automatically on GPU out-of-memory |
| `--num-workers N` | integer ≥ 0 | `encode.num_workers` (2) | DataLoader processes reading volumes from the cache (0 = read in the main process) |
| `--min-free-gb GB` | number ≥ 0 | `encode.min_free_gb` (20) | refuse to start, or stop cleanly, below this much free disk on the store's filesystem (0 = no check) |
| `--limit N` | integer ≥ 1 | `encode.limit` (null = all) | encode at most the first N selected volumes (smoke tests) |
| `--dry-run` | flag | off | only report what is selected, what is already encoded, and the store size; writes nothing |

Not command-line arguments (edit the YAML): `selection.qc_passed_only` (default true) and
`encode.embeddings_dir` (default `data/embeddings`) in encode.yaml; `data_config:` (the preprocessing
config that gives `cache_dir` / `runs_dir`); and everything about the backbone itself (weights, input
transform, input size, pooling, store dtype), which is in `configs/model/encoders/<name>.yaml`.

### `scripts/model/check_encoder.py`

Writes nothing. Selection (splits, QC, kernel classes) comes from encode.yaml.

| Argument | Values | Default | What it does |
|---|---|---|---|
| `--config PATH` | a YAML file | `configs/model/encode.yaml` | the stage-2 settings file |
| `--encoder NAME` | as above | `encoder:` in encode.yaml | which backbone to test |
| `--source NAME` | as above | `selection.source` | whose runs to read |
| `--run NAME` | as above | `selection.run` | the run to take the test volume from, and to project the total time for |
| `--volume-id ID` | a selected `volume_id` of the run | the first selected volume | which volume to encode |
| `--device D` | `auto` `cuda` `cpu` | `encode.device` | as above |
| `--amp-dtype T` | `auto` `bfloat16` `float16` `float32` | `encode.amp_dtype` | as above |
| `--slice-batch-size N` | integer ≥ 1 | `encode.slice_batch_size` | the batch size to measure speed and peak VRAM with |
| `--max-slices N` | integer ≥ 0 | 64 | slices to encode, evenly spread along the body axis (0 = the whole volume) |

## What gets selected

Rows of `data/runs/<source>/<run>/manifest.csv` with `split` in `selection.splits` (default
train/val/test, never `excluded` / `unassigned`), `qc_passed` true, and `kernel_class` in
`selection.kernel_classes` (default: all). Cache files are found as `<cache_dir>/{volume_id}.npy`.
The manifest's `npy_path` column is not used, because it keeps the separators of the machine that
merged it. A selected volume missing from the cache is an error, never a silent skip.

## Disk, memory, time

| | |
|---|---|
| Store size | ~0.47 MB per volume (230 slices × 1024 × 2 B); **10k volumes ≈ 4.7 GB** |
| Disk guard | stops cleanly below `encode.min_free_gb` (20 GB) free; it checks before loading the model, then before every volume |
| GPU memory | ViT-L bf16 + `slice_batch_size` slices; on out-of-memory the batch halves automatically and stays halved |
| Host memory | one ~25 MB int16 volume per DataLoader worker in flight |
| Time | about `total slices / slices per second` (both printed by `check_encoder.py`); expect hours, not days, for 10k volumes on the GPU. On a laptop CPU, ViT-L runs at ~1 slice/s: only for smoke tests |

## Adding a backbone

1. Copy `configs/model/encoders/dale_ct_2s.yaml` to `configs/model/encoders/<name>.yaml`. Set `arch`
   and `model_args` (any timm ViT), `weights` (a HF safetensors file at a pinned commit, or
   `timm_pretrained`), the **input transform the backbone was trained with** (`clip_zscore` for one
   z-scored channel, `multi_window` for windowed RGB plus normalisation), `pooling` and `embed_dim`.
2. Run `check_encoder.py --encoder <name>`. Weights load strictly: a key mismatch is an error naming
   the keys (allow deliberate ones with `weights.ignore_prefixes`).
3. Run `encode_volumes.py --encoder <name>`. Its embeddings go to a new folder; earlier stores are untouched.

A non-timm family registers a builder with `@register_encoder("<type>")` in `src/ct_model/encoders/`
and sets `type: <type>` in its YAML (see `registry.py`).

## Stages 3-4: train, evaluate, compare

Phase 1 trains **stage 3 (ABMIL aggregation) and stage 4 (linear head, 18 labels)** on the frozen
embeddings. The encoder is not trained. The design (equations, imbalance, evaluation protocol) is in
[architecture.md](architecture.md#stages-3-4-aggregation-and-classification-phase-1).

### 0. Embeddings for the run (once)

```bash
python scripts/model/encode_volumes.py --run train-sharp-5800 --dry-run
```
```bash
python scripts/model/encode_volumes.py --run train-sharp-5800 2>&1 | tee -a encode_train-sharp-5800.log
```
Run the second command in tmux (see stage 2 above). Every usable volume of the run gets embeddings:
train, val and test, both kernels; 10,042 volumes, ~4.7 GB. Training refuses to start if any selected
volume is missing.

### 1. Train, in tmux

```bash
tmux new -s train
```
```bash
python scripts/model/train_mil.py --experiment abmil_dale2s 2>&1 | tee -a train_abmil_dale2s.log
```
Detach with `Ctrl+b d`; `tmux a -t train` reattaches.
- **First line:** the settings actually used.
- **Each epoch:** train / val loss and val macro AUROC / AUPRC; `*best` marks a new best epoch.
- **Interrupted?** Run the same command again: it resumes from the last finished epoch.
- **Finished?** That config + seed is refused; nothing is ever overwritten.

Training reads **train and val only**. At the end it writes the val predictions and the per-label
thresholds (chosen on val), and adds a row to `outputs/experiments/index.csv`.

**How long.**
- **Steps:** an epoch is `ceil(5,827 / 16) = 365` steps. The budget is 100 epochs (36,500 steps), and
  training stops after 15 epochs without improvement.
- **Measure, don't assume:** look at the best epoch in the report notebook. If it is in the last 10 %
  of the budget, raise `--max-epochs`.

### 2. Compare on val (several configurations × 5 seeds)

Same tmux pattern, one command per run. Seeds of one configuration share a config hash and are averaged:
```bash
for s in 0 1 2 3 4; do python scripts/model/train_mil.py --experiment abmil_dale2s --seed $s; done
```

**Recommended first comparisons.** Each changes one thing against `abmil_dale2s` (BCE, single attention):

| Question | Command change |
|---|---|
| Does attention beat a plain mean? | `--experiment meanpool_dale2s` |
| One attention per label? | `--experiment abmil_perlabel_dale2s` |
| Does the loss handle imbalance better? | `--loss weighted_bce`, `--loss asl` |

Then summarise the ledger (mean ± std per configuration):
```bash
python scripts/model/summarize_experiments.py
```
Pick on **val** macro AUPRC / AUROC. Test is not looked at yet.

### 3. Evaluate on test, once per chosen run

```bash
python scripts/model/evaluate_mil.py --experiment-dir outputs/experiments/abmil_dale2s/<hash8>/seed0
```
- **What it uses:** the best checkpoint, with the thresholds chosen on val, unchanged.
- **What it reports:** overall, **per kernel class** (sharp vs soft is the cross-kernel result) and
  **per scanner**, with patient-level bootstrap 95 % CIs.
- **Where:** results go to `eval/ctrate__train-sharp-5800__test/` in the run folder, plus a row in
  `outputs/experiments/evaluations.csv`.
- **Repeats are refused.** `--overwrite` redoes an evaluation deliberately.
- **Then the clinical report** of that evaluation is written (next section). `--no-clinical` skips it.

`summarize_experiments.py --split test` compares the evaluated runs.

### 3b. Clinical evaluation: is it usable? (every phase)

AUROC says how well scans are *ranked*. The clinical report says what a reader of the output would get, so every
phase (frozen MIL, LoRA, slice selection, NHRD) is judged on the same yardstick. It reads **only saved
predictions** (`predictions_val.csv` and `eval/<...>/predictions.csv`), never a model: anything that writes those
files is scored the same way.

```bash
python scripts/model/clinical_report.py --experiment-dir outputs/experiments/abmil_dale2s/<hash8> --per-seed
```
- **A config folder** (`<name>/<hash8>`) scores the **seed ensemble** (probabilities averaged over its evaluated
  seeds) into `<hash8>/ensemble/clinical/<eval>/`; `--per-seed` adds every seed's own report. A run folder
  (`.../seed<k>`) scores that run alone into `seed<k>/clinical/<eval>/`. `evaluate_mil.py` already writes the run's
  own report, so this is for ensembles, older evaluations, and re-scoring after changing the settings.
- **Chosen on val, applied unchanged:** per label, a calibration (Platt) and two thresholds:
  - **present** when p ≥ the *report* threshold (val specificity ≥ 0.95);
  - **absent** when p < the *rule-out* threshold (val sensitivity ≥ 0.90);
  - **uncertain** in between: a human looks.

  The phase-1 max-F1 threshold is kept as `maxf1_*` for comparison.
- **What it reports:**
  - **per scan:** false labels and missed findings, exact match, normal scans flagged, uncertain labels;
  - **per label:** AUROC with patient-bootstrap CI, AUPRC, sensitivity / specificity / PPV / NPV at both
    thresholds, sensitivity at fixed specificities, calibration error before and after calibration, Brier score,
    decision-curve net benefit;
  - **kernel robustness:** the same scan reconstructed sharp and soft (test holds both): how often a decision
    changes, and the soft − sharp sensitivity gap;
  - everything also per kernel class and per scanner.
- **A gate per label:** **supported** (passes every rule), **review** (fails only calibration or robustness),
  **not supported** (fails discrimination: AUROC CI low, or sensitivity at the report threshold). Rules and
  targets are in `configs/model/clinical.yaml`; change them there, never in code.
- **Where:**
  - in the report folder: `scorecard.json` (headline + label status: read it first), `per_label.csv`,
    `summary.csv`, `per_scan.csv`, `paired_kernel.csv`, `paired_scans.csv`, `gate.csv`,
    `operating_points.json`, `predictions_clinical.csv` (calibrated probabilities and the 3-state output);
  - one row per report in `outputs/experiments/clinical.csv`.

  A report is derived and deterministic, so running it again rewrites it.
- **Look at it:** `notebooks/clinical_report.ipynb` compares every config's scorecard and label status, both rules,
  the kernel robustness, calibration (sharp vs soft), decision curves, false labels per scan, and the gate for
  moving on to slice selection.

**Encoder diagnostics** (also every phase, per embedding store): section 7 of
`notebooks/encoder_embeddings_report.ipynb`, from `ct_model.embeddings.diagnostics`.
- **Kernel probe:** how readable sharp vs soft is from the mean embedding (frozen DALE: 0.9999; adaptation
  should push it down).
- **Linear probe:** macro AUROC of a logistic regression on the mean embedding (frozen DALE: 0.769), the floor
  any aggregator must beat.

### Several runs in one experiment

`data.run` can join runs of one source with `+`, e.g. `train-sharp-5800+train-soft-1000`
(`configs/model/experiments/abmil_perlabel_mixed_dale2s.yaml`).
- **How they are joined:** the manifests are concatenated; a volume listed by several runs (the shared test pool)
  is kept once.
- **No leakage:** a patient must have the same split in every run, which holds for a run built with
  `make_worklist.py --from-run`. Otherwise it is an error.
- **Config hash:** covers every part's manifest.

The same name works for `encode_volumes.py --run` and `evaluate_mil.py --run`.

### 4. Look at it

Open `notebooks/experiments_report.ipynb` and *Run all*. It reads every run folder under
`outputs/experiments/` (no folder to pick): an inventory of all runs, every config averaged over its seeds on
val and on test (overall / sharp vs soft / per scanner), the training curves of all configs together (budget
and over-fitting check), per-finding AUROC against a reference config, and one run in detail. The loading is
in `ct_model.training.results`, for use elsewhere.

### Where results are, and how they are identified

```
outputs/experiments/
  index.csv                        every finished training run: run_id, config hash, key settings, seed,
                                   git commit, best epoch, val metrics, folder
  evaluations.csv                  every evaluation: run_id, split, run, macro AUROC (+ CI), per-kernel AUROC
  <name>/<config hash8>/seed<k>/   one training run (layout in architecture.md)
```

- **Run id:** `<name>/<hash8>/seed<k>`.
- **What the config hash covers:** everything that changes what is learned:
  - the resolved config (without name, seed, device and paths);
  - the embedding store (encoder, weights commit, preprocessing);
  - the run manifest (volumes, splits, labels).
- **Using it:** two runs with the same hash are the same experiment with different seeds. Keep
  `outputs/` (it is git-ignored) and the ledgers make any later comparison possible.

### Arguments

#### `scripts/model/train_mil.py`

Defaults come from `configs/model/experiments/<experiment>.yaml`; each argument overrides one key. Any
change to a hashed setting creates a new config hash, so the run goes to a new folder.

| Argument | Values | Default (YAML key) | What it does |
|---|---|---|---|
| `--experiment NAME` | a file in `configs/model/experiments/` (without `.yaml`), or a path | required | the experiment config |
| `--run NAME` | a run of the source | `data.run` (`train-sharp-5800`) | whose manifest gives splits and labels |
| `--seed N` | integer | `seed` (0) | seed for initialisation, shuffling, dropout; not part of the config hash |
| `--loss T` | `bce` `weighted_bce` `asl` | `loss` (`bce`) | loss type; ASL keeps its reference defaults |
| `--lr X` | > 0 | `optim.lr` (2e-4) | peak learning rate (AdamW) |
| `--weight-decay X` | ≥ 0 | `optim.weight_decay` (0.01) | weight decay on weight matrices |
| `--batch-size N` | ≥ 1 | `optim.batch_size` (16) | volumes per step |
| `--max-epochs N` | ≥ 1 | `optim.max_epochs` (100) | epoch budget |
| `--patience N` | ≥ 1 | `optim.patience` (15) | epochs without improvement before stopping |
| `--device D` | `auto` `cuda` `cpu` | `device` (`auto`) | where to train |
| `--output-dir PATH` | folder | `output_dir` (`outputs/experiments`) | where runs and ledgers go |

Settable only in the YAML:
- `data.train_kernel_classes`, `data.labels`, `data.train_slice_sampler`, `data.preload`;
- `features.standardize`;
- the aggregator's `type`, `hidden_dim`, `attn_dim`, `dropout`, `branches`;
- the head's `type` and `dropout`, and `head_prior_bias`;
- loss parameters (e.g. ASL's `gamma_neg`);
- `optim.warmup_epochs`, `grad_clip`, `monitor`.

#### `scripts/model/evaluate_mil.py`

| Argument | Values | Default | What it does |
|---|---|---|---|
| `--experiment-dir DIR` | a finished run folder | required | the run to evaluate |
| `--split S` | `train` `val` `test` | `test` | split to score |
| `--source NAME` | a source | the experiment's | to score another source (same labels) |
| `--run NAME` | a run | the experiment's | to score another run (e.g. a soft subset) without retraining |
| `--kernel-classes K [K ...]` | e.g. `soft` | all | only these kernel classes |
| `--bootstrap N` | ≥ 0 | 1000 | bootstrap resamples (patients) for the 95 % CIs; 0 = none |
| `--min-group N` | ≥ 1 | 100 | smaller groups are flagged `too_small` and get no CI |
| `--device D` | `auto` `cuda` `cpu` | `auto` | where to predict |
| `--overwrite` | flag | off | redo an existing evaluation |
| `--no-clinical` | flag | off | skip the clinical report written after the evaluation |

#### `scripts/model/clinical_report.py`

| Argument | Values | Default | What it does |
|---|---|---|---|
| `--experiment-dir DIR` | a run folder, or a config folder | required | one run, or the seed ensemble of a config |
| `--split S` | `train` `val` `test` | `test` | which evaluation to report |
| `--source NAME` / `--run NAME` | a source / a run | the experiment's | the evaluation of another source or run |
| `--clinical-config PATH` | YAML | `configs/model/clinical.yaml` | operating-point targets, calibration, gate rules |
| `--per-seed` | flag | off | with a config folder, also every seed's own report |
| `--bootstrap N` | ≥ 0 | from the YAML (1000) | patient resamples for the per-label AUROC CIs |

#### `scripts/model/summarize_experiments.py`

| Argument | Values | Default | What it does |
|---|---|---|---|
| `--output-dir PATH` | folder | `outputs/experiments` | where the ledgers are |
| `--split S` | `val` `test` | `val` | val results (`index.csv`) or evaluations (`evaluations.csv`) |
| `--name NAME` | experiment name | all | only this experiment |

#### `scripts/model/explain_volume.py`

| Argument | Values | Default | What it does |
|---|---|---|---|
| `--experiment-dir DIR` | a finished run folder | required | the trained model |
| `--volume-id ID` | a usable volume of the run | required | the volume to explain |
| `--labels L [L ...]` | label names | labels predicted positive at the val thresholds | which labels |
| `--top-k N` | ≥ 1 | 5 | top slices shown per label |
| `--heatmaps` | flag | off | also in-slice Grad-CAM maps (re-encodes the top slices: use a GPU) |
| `--min-cosine X` | 0-1 | 0.99 | required agreement between re-encoded and stored embedding |
| `--run NAME` | a run | the experiment's | the run holding the volume |
| `--device D` | `auto` `cuda` `cpu` | `auto` | where to run |

## Phase 2: adapt the encoder (LoRA)

The frozen encoder limits results:
- a linear probe on its mean embedding reaches 0.769, and every aggregator reaches about 0.80;
- the reconstruction kernel is readable from its embeddings at AUROC 0.9999.

Phase 2 trains LoRA adapters inside the encoder against this. Details are in `src/ct_model/adapt/` and
`src/ct_model/encoders/lora.py`.

```bash
python scripts/model/adapt_encoder.py --adapt-config lora_dale2s_kernel --init-from outputs/experiments/<name>/<hash8>/seed0
```

**What trains.**
- LoRA (r = 8) on `attn.qkv` and `attn.proj` of the last 12 of the 24 blocks. That is 0.6 M parameters; the
  ViT-L weights stay frozen.
- The aggregator and head, initialised from `--init-from`. Give a phase-1 run trained on the same data run.

**How a volume label trains the encoder.** No slice labels are needed:
- The volume loss backpropagates through the attention to every sampled slice.
- 48 slices per volume go through the encoder with gradients (`random_k`, one per segment along the body axis).
- The first 12 blocks run without gradients and the adapted ones are checkpointed, so it fits in 16 GB.

**Kernel consistency.** Training items are *scans*: a scan with a sharp and a soft reconstruction (about
1,000 in `train-sharp-5800+train-soft-1000`) brings both. Their slices are aligned in z by the per-slice
mean-HU profile. Two label-free terms are added to the BCE:
- the aligned slices should embed alike: 1 − cosine after FeatureNorm (frozen DALE: 0.61, so the loss is
  about 0.39);
- the two volumes should predict alike: symmetric KL.

A one-kernel scan gets a blurred (sharp) or sharpened (soft) copy of the same slices with probability
`augment_prob`.

**Selection.** Each epoch is scored on val: 48 uniform slices, both kernels.
- Logged in `metrics.csv`: macro AUROC (overall, sharp, soft), val loss, and for the val pairs the slice
  cosine, |Δp| and decision flips at 0.5. Epoch 0 is the frozen start.
- The best epoch on `optim.monitor` is kept.

**Output.** `outputs/adapters/<name>/<hash8>/seed<k>/`:
- `adapter.safetensors`;
- `encoder.yaml`: the base encoder plus `lora.adapter`, a new encoder with its own fingerprint and store;
- `experiment.yaml`: the init run's experiment on that encoder;
- `outputs/adapters/index.csv`: the ledger.

The same command resumes an interrupted run.

**Then, with the adapted encoder frozen again,** the usual pipeline: the head trains on full bags, so
training and test see the same bag size.

```bash
python scripts/model/encode_volumes.py --encoder outputs/adapters/<name>/<hash8>/seed0/encoder.yaml --run train-sharp-5800+train-soft-1000
```
```bash
python scripts/model/train_mil.py --experiment outputs/adapters/<name>/<hash8>/seed0/experiment.yaml --seed 0
```

Then evaluate each seed, run the clinical report of the ensemble, and run section 7 of the encoder
notebook with `ENCODER` set to that `encoder.yaml`. Section 8 of `clinical_report.ipynb` decides whether
it passes the gate for slice selection.

**Budget on the 4070 Ti (16 GB).**
- About 6.8k volume views of 48 slices per epoch; the time per epoch is in the log.
- Val takes a few minutes per epoch.
- The adapted store takes about 80 minutes to encode.

## Experiment tracking: MLflow / DagsHub

`outputs/` stays the source of truth. `scripts/model/export_mlflow.py` publishes a shareable view of it to an
MLflow server, for example the project's DagsHub repository.

The export is idempotent: a run already on the server is recognised by its `ct.run_id` tag and skipped. Run it
after every phase:

```bash
python scripts/model/export_mlflow.py --dry-run
```
```bash
python scripts/model/export_mlflow.py
```

**Login.** Use the standard MLflow variables, set in your shell or in `~/.bashrc`. Never put them in the repo:
- `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`, a DagsHub access token.

**MLflow layout.**

| MLflow experiment | MLflow runs | Source folder |
|---|---|---|
| one per experiment name | `<hash8>/seed<k>` | `outputs/experiments/<name>/<hash8>/seed<k>/` |
| the same experiment | `<hash8>/ensemble` | the seed-ensemble clinical report |
| `adapt/<name>` | one per LoRA run | `outputs/adapters/<name>/<hash8>/seed<k>/` |

**Params:** the flattened config.

**Metrics:**
- the per-epoch curves;
- final val metrics;
- test metrics: `test.macro_auroc`, `test.macro_auroc.sharp`, `test.label.<finding>.auroc`;
- the clinical headline: `clinical.test.report_fp_per_scan`, `clinical.test.paired_scans_with_flip`, ...;
- an evaluation on another data run (e.g. NHRD) is keyed `<source>.<run>.<split>`.

**Tags:** git commit, config hash, the label status (`clinical.test.supported`).

**Artifacts:** configs, curves, evaluation and clinical tables, and the weights (`best.pt`; LoRA
`adapter.safetensors` + `head.pt`).

**Never exported:**
- the HU cache and embedding stores;
- resume checkpoints;
- per-volume prediction files: they carry CT-RATE patient-level data and labels (CC BY-NC-SA). Only with
  `--with-predictions`, and only to a private repository.

`--refresh` re-sends runs after a new evaluation or clinical report. `--name` limits the export to one
experiment.

## Evidence (optional)

Nothing in training depends on it. To see why the model predicted a label for one volume:

```bash
python scripts/model/explain_volume.py --experiment-dir outputs/experiments/abmil_dale2s/<hash8>/seed0 --volume-id valid_1127_a_1 --heatmaps
```
Or use `notebooks/evidence_viewer.ipynb`, which calls the same functions interactively.

**Which slices (L1).** The exact contribution of every slice to the label's logit: positive pushes
towards "present". These come from the trained model alone, with no approximation.

**Where in the slice (L2, `--heatmaps`).** A Grad-CAM map over the top slices. The encoder is rebuilt
from the embedding store's own record, and its re-encoded slice must match the stored embedding.

Output goes to `<run>/explain/<volume_id>/`: one PNG per label plus `evidence.json` (probabilities,
val thresholds, top slices and their contributions).

**Before trusting the heatmaps**, run the two sanity checks in the notebook (section 3) on a few
volumes:
- randomising the head must change the map;
- blanking the top patches must lower the logit more than blanking random ones.

**Limits.**
- **Not a localisation:** this is what the model relied on, not a validated localisation of the
  finding.
- **Slice numbers:** they are slices of the preprocessed volume (1.5 mm, foot to head, body-cropped),
  not original DICOM slice numbers.

## Troubleshooting

| Message | Meaning / fix |
|---|---|
| no progress lines in the terminal or log for a long time | an old checkout: every `scripts/model` script now line-buffers its output, so `\| tee` shows each line as it happens. For an already running old process, count the outputs instead (e.g. `ls <store>/*.npy \| wc -l`), or restart it with `python -u` (it resumes) |
| `... volume(s) have no embeddings in ...` | run `encode_volumes.py --run <run>` first (all splits) |
| `... is already trained` | that config + seed is finished; use another `--seed`, change a setting, or evaluate it |
| `... this evaluation was already done` | results exist; `--overwrite` only if redoing it is intended |
| `val val_macro_auroc cannot be computed` | no label has both classes in val: the val split is too small |
| `re-encoded slice ... has cosine ...` | the encoder / weights / volume differ from what the classifier saw; heatmaps refused |
| `StoreMismatch ... HU cache fingerprint` | the cache was rebuilt with other preprocessing settings since these embeddings were made. Delete that store folder and re-encode |
| `StoreMismatch ... weights commit` | the encoder YAML now points at other weights. Revert, or delete the store folder |
| `checkpoint ... does not match the model` | `arch` / `model_args` do not match the weights file |
| `not a multiple of ... patch size` | `input.size_hw` must be divisible by the patch size (16 for DALE) |
| `device cuda was asked for but torch sees no GPU` | a CPU-only torch is installed, or the driver is missing |
| `only N GB free ... below min_free_gb` | free disk (the HU cache is large), or lower `--min-free-gb` deliberately |
| `the first 5 volumes all failed` | something systematic (weights, device, config); the first error is shown |

## Development

```bash
python -m pytest tests/model
```
The tests use synthetic volumes, a synthetic embedding store with planted findings, and a tiny random
ViT, with no download and no GPU. What they verify against a reference:

| Component | Checked against |
|---|---|
| DALE input transform | the model card's `CTInferenceTransform` code |
| ABMIL | Eq. 7 and Eq. 9 of Ilse et al. 2018, computed by hand; padding and permutation invariance |
| `weighted_bce` / `bce` | `torch.nn.BCEWithLogitsLoss` (with `pos_weight`) |
| `asl` | an independent re-implementation of the authors' reference code, values and gradient |
| metrics | scikit-learn; max-F1 thresholds by brute force; bootstrap draws whole patients |
| slice evidence (L1) | logits = sum of contributions + bias, exactly; top slice = planted slice (≥ 90 %) |
| heatmaps (L2) | an independent autograd computation of Grad-CAM |
| training / evaluation | learns a planted signal (val AUROC > 0.9), never reads test, resumes, never overwrites |
