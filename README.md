# Sound Matching with a Differentiable Karplus-Strong Algorithm

Official code for the DAFx26 paper, by
[Pablo Tablas](https://github.com/ptablasdpaula) (<p.tablasdepaula@qmul.ac.uk>).

Audio examples, figures, and an interactive demo:
https://ptablasdpaula.github.io/DAFx26-Karplus/

Plucked-instrument sound matching is framed here as event-based parameter
estimation. An encoder predicts discrete pluck events that drive a differentiable
extended Karplus-Strong decoder, in either a time-domain (`tKSA`) or a
frequency-sampling (`fKSA`) implementation. Three training regimes are compared:

| Objective    | Supervision                                        |
|--------------|----------------------------------------------------|
| `p_only`     | parameter supervision only                          |
| `p_audio`    | parameter supervision plus audio losses              |
| `audio_only` | audio losses with external onset/f0 detectors        |

The Harmonics-plus-Noise baselines are `hpn` and `hpn_p`.

---

## 1. Installation

Install [Pixi](https://pixi.sh), then create the CPU environment:

```bash
pixi install
```

Training and evaluation need a CUDA GPU. Build that environment on a GPU node:

```bash
CONDA_OVERRIDE_CUDA=12.9 PIXI_CACHE_DIR=/tmp/pixi-$USER MAX_JOBS=2 pixi install -e cuda
CUDA_HOME="$PWD/.pixi/envs/cuda" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
  .pixi/envs/cuda/bin/python -m pip install --force-reinstall --no-deps \
  --no-build-isolation --no-cache-dir torchlpc==0.7.2
```

Pointing `PIXI_CACHE_DIR` at node-local storage avoids incomplete rattler cache
extraction on shared filesystems.

Then create your local config:

```bash
cp .env.example .env
```

`.env` holds everything site-specific — dataset location, cluster settings, and
an optional Weights & Biases key. It is gitignored and never leaves your machine.
Every stage below reads it.

## 2. Data

Synthetic data is generated on the fly by `data/synthetic_dataset.py`; nothing to
download. The real-guitar experiments use the NSynth acoustic-guitar subset.

Set where the dataset should live in `.env` (any path — scratch, an external
drive; leave blank to use `./data/nsynth`):

```dotenv
NSYNTH_DIR=/path/to/nsynth
```

Then download and preprocess:

```bash
python data/nsynth/download.py        # ~70 GB of archives, expanded in place
data/nsynth/run_preprocessing.sh      # onset, f0, and loudness detection
```

Preprocessing runs CREPE over every item and wants a GPU, so it submits to SLURM
using the settings from stage 3. On a local GPU machine, run the payload
directly instead:

```bash
pixi run -e cuda python data/nsynth/preprocess_subset.py
```

This writes `preprocessed/` inside each split plus a shared
`loudness_stats.json`.

## 3. Training

Fill in the SLURM block of `.env` for your site. Blank values are never passed to
`sbatch`, so leave anything your cluster does not use empty:

```dotenv
SM_SLURM_ACCOUNT=your-account
SM_SLURM_PARTITION=your-partition
SM_SLURM_QOS=
SM_SLURM_GRES=gpu:1            # or blank + SM_SLURM_GPUS=1 on newer sites
SM_SLURM_CONSTRAINT=a100
SM_SLURM_PREAMBLE='module load cuda/12.9'
```

`SM_SLURM_PREAMBLE` runs inside the allocation before python starts, which is
where module loads and any other site setup belong. Logging is offline by
default; set `WANDB_MODE=online` and your own `WANDB_API_KEY`/`WANDB_ENTITY` to
log to your workspace.

Submit the full matrix from the paper:

```bash
experiments/jobs/run_all_training.sh
```

The matrix is one line of Hydra overrides per run in
`experiments/jobs/experiments.conf`. To submit a single row, pass that line to
`run_training.sh`:

```bash
experiments/jobs/run_training.sh \
  data=mix detector=end_to_end training=combined model=ksa_freq
```

Or run one directly on a local GPU, no scheduler involved:

```bash
pixi run -e cuda python experiments/train.py \
  data=mix detector=end_to_end training=combined model=ksa_freq
```

## 4. Evaluation

To reproduce the paper's numbers without training anything, fetch the trained
checkpoints. They live on the `checkpoints` branch, so cloning the code does not
drag ~140 MB of weights along:

```bash
scripts/fetch_checkpoints.sh
```

Then recompute Tables 3–4 and render the audio examples:

```bash
experiments/jobs/run_all_evaluation.sh          # via SLURM
pixi run -e cuda python experiments/evaluate.py --mode all   # or directly
```

Outputs:

```text
experiments/evaluation/synthetic_results.csv
experiments/evaluation/real_results.csv
experiments/evaluation/audio/{synthetic,real}/
```

The reference CSVs from the paper are committed, so you can diff your run against
them. A subset works too:

```bash
pixi run -e cuda python experiments/evaluate.py --mode nsynth \
  --tags real_fKSA_p_audio,real_tKSA_p_audio --out_csv /tmp/subset.csv
```

Note that `--mode nsynth` selects the real-guitar dataset loader, while the
paper-facing output paths say `real`.

---

## Repository Layout

```text
src/                      Model, encoder, decoders, detectors, losses, metrics.
src/synths/               Karplus-Strong DSP/DDSP implementations.
data/                     Synthetic dataset, NSynth download and preprocessing.
experiments/              Training, evaluation, and gradient analysis.
experiments/configs/      Hydra config groups: model, detector, training, data.
experiments/jobs/         SLURM wrappers and the experiment matrix.
figures/                  Notebooks for the paper's generated figures.
scripts/                  Checkpoint fetch.
```

## Other Branches

| Branch                          | Contents                                              |
|---------------------------------|-------------------------------------------------------|
| `checkpoints`                   | Trained weights for every run in the paper (Git LFS).  |
| `web`                           | Project page, figures, and rendered audio examples.    |

The full pre-release development history, including superseded checkpoints, is
archived at [`DAFx26-Karplus-archive`](https://github.com/ptablasdpaula/DAFx26-Karplus-archive).
