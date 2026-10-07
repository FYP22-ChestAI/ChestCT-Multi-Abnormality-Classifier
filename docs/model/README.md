# Model pipeline: stage 2 (slice encoder)

Stage 2 turns every volume of a run into per-slice embeddings. Each volume's `(N, 224, 224)` int16 HU
in `data/cache/` becomes an `(N, 1024)` float16 array in `data/embeddings/<encoder>-<fp>/`. That
array is what stages 3 (aggregation) and 4 (classifier) train on in phase 1. How the stages fit
together, the store format and the roadmap are in [architecture.md](architecture.md).

The HU cache is the source of truth and is only read. The embedding store is derived: delete it and
re-run to rebuild it.

## Setup (once, on the server)

```bash
python -m pip install -e ".[dev,model]"
```
Installs torch, timm and safetensors with the package. Use a CUDA build of torch on the server (see
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

**3. Encode.** It is resumable: stop it any time (Ctrl-C, a lost SSH session) and run the same command
again. Volumes already in the store are skipped, and every file is written atomically.
```bash
nohup python scripts/model/encode_volumes.py --run train-sharp > encode_train-sharp.log 2>&1 &
```
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

## Troubleshooting

| Message | Meaning / fix |
|---|---|
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
The tests use synthetic volumes and a tiny random ViT, with no download and no GPU.
`test_dale_transform_equals_model_card` checks our input transform against the model card's code.
