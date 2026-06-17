# Sound Matching with a Differentiable Karplus-Strong Algorithm
This is the official accompanying repository for the DAFx26 paper.

🔊 **Audio examples & project page:** https://ptablasdpaula.github.io/DAFx26-Karplus/

We frame plucked-instrument sound matching as **event-based parameter estimation**: an
encoder predicts a set of discrete pluck events (onset, $f_0$, timbre) that drive a
**differentiable extended Karplus-Strong** decoder — time-domain (`tKSA`) or
frequency-sampling (`fKSA`). We compare three training regimes — parameter-loss only
(`P-Only`), audio losses with external detectors (`Audio-Only`), and joint
(`P+Audio`), with stop-gradient ablations — against Harmonics-plus-Noise baselines.

The repo is organised as such:
```
figures/                  <- Notebooks that generate the figures shown in the paper.
data/                     <- NSynth filtering/preprocessing and synthetic-data generation.
src/                      <- Encoder, decoders, model, detectors and losses.
src/synths                <- Karplus-Strong DSP and DDSP implementations.
experiments/              <- Gradient analysis, training and sound-matching evaluation.
experiments/checkpoints/  <- Trained model checkpoints used in the paper.
experiments/evaluation/   <- Result CSVs and the audio examples served by the project page.
experiments/jobs/         <- SLURM job scripts (env build, preprocessing, training, eval).
```

## Installation

### Local (CPU)
1. Install [Pixi](https://pixi.sh): `brew install pixi` (or the official installer script).
2. Clone and install:
```bash
git clone https://github.com/ptablasdpaula/DAFx26-Karplus.git
cd DAFx26-Karplus
pixi install            # default (cpu) environment
pixi shell
```

### GPU cluster (Apocrita)
The CUDA environment compiles native extensions (`philtorch`, `torchlpc`) and is
sensitive to a few cluster quirks, so **build it from a GPU compute node**, not the
login node:

```bash
sbatch experiments/jobs/build_env.sh     # builds the `cuda` env + runs a GPU smoke test
```

`build_env.sh` encodes the three things a manual `pixi install` gets wrong here:

1. **Cache on node-local disk** (`PIXI_CACHE_DIR=/tmp/...`). The rattler cache
   extracts incompletely on the shared gpfs filesystem ("could not open source
   file" link errors). Local `/tmp` is reliable.
2. **Capped compile parallelism** (`MAX_JOBS=2`, also pinned in `pixi.toml`).
   philtorch's `csrc/*.cpp` need several GB each; unbounded `ninja` OOM-kills the
   compiler. Login nodes also cap per-user memory, hence building on a compute node.
3. **Rebuild `torchlpc` with its compiled CUDA kernels.** torchlpc 0.7.2 *does*
   ship CUDA kernels (`csrc/cuda/{linear_recurrence,lpc}.cu`) that register
   `torchlpc::scan`/`lpc` for CUDA, but `setup.py` only compiles them when
   `CUDA_HOME` is set at build time — and pixi's build subprocess leaves it unset,
   producing a **CPU-only** `_C`. With a CPU-only `_C` present, `recurrence.py`
   still routes CUDA tensors to it → `torchlpc::scan` has no CUDA backend and the
   time-domain synth crashes on GPU. Rebuilding from source with `CUDA_HOME` set
   includes the optimised compiled CUDA scan (the intended fast path).

If you install manually instead of via the job (run on a GPU node):
```bash
CONDA_OVERRIDE_CUDA=12.9 PIXI_CACHE_DIR=/tmp/pixi-$USER MAX_JOBS=2 pixi install -e cuda
CUDA_HOME="$PWD/.pixi/envs/cuda" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
  .pixi/envs/cuda/bin/python -m pip install --force-reinstall --no-deps \
  --no-build-isolation --no-cache-dir torchlpc==0.7.2
# verify: nm -D .pixi/envs/cuda/lib/python*/site-packages/torchlpc/_C*.so | grep scan_cuda_wrapper
```

Run scripts call `.pixi/envs/cuda/bin/python` directly (not `pixi run`) so the
manual `_C` removal is not undone by a re-sync.

## Reproducing the paper

All commands assume the environment is installed (above). On the cluster, run the
`sbatch` job scripts; locally, call the python entry points directly. Scripts invoke
`.pixi/envs/cuda/bin/python` so the compiled `torchlpc` CUDA path is preserved.

> **Quick start — reproduce Tables 3–4 without retraining.** The trained checkpoints
> ship in `experiments/checkpoints/`, so you can go straight to evaluation:
> ```bash
> python experiments/evaluate.py --mode all   # writes experiments/evaluation/{synthetic,nsynth}_results.csv
> ```
> Steps 1–3 below are only needed to regenerate the data, gradient analysis, or
> retrain the models from scratch.

### 1. Data
- **Synthetic** data is generated on the fly from `data/synthetic_dataset.py` (seeded
  per epoch); nothing to download.
- **NSynth** (real guitar) — download and filter the acoustic-guitar subset:
  ```bash
  python data/nsynth/download.py            # fetch NSynth
  bash   data/nsynth/run_preprocessing.sh   # filter to D1–D6, build train/val/test splits
  # cluster equivalent: sbatch experiments/jobs/preprocess_nsynth.sh
  ```

### 2. Gradient analysis (Tables 1–2, single/multi-event)
The notebooks produce the accuracy CSVs (already committed under `experiments/`):
```bash
experiments/gradient_analysis.ipynb                 # single-event CGA/FGA  -> gradient_accuracy_single_sw.csv
experiments/gradient_analysis_sliced_wasserstein.ipynb   # multi-event Any/Joint -> gradient_accuracy_multi_event*.csv
```

### 3. Training the models
Training uses Hydra. Config groups: `model={ksa_time,ksa_freq,ksa_oracle,hn,hn_tcn}`,
`data={synth,real,mix}`, `detector={end_to_end,external}`, `training={combined,param_only,spectral_only}`.
The audio-loss weights follow Torres et al. (`training.w_mss=0.05 training.w_sot=1.0`),
with `training.w_param=1.0` for joint training.

```bash
# P-Only (parameter loss on synthetic data)
python experiments/train.py model=ksa_oracle data=synth detector=end_to_end training=param_only
# P+Audio (joint), time-domain decoder
python experiments/train.py model=ksa_time data=synth detector=end_to_end training=combined \
       training.w_mss=0.05 training.w_sot=1.0 training.w_param=1.0
# Audio-Only (external detectors), real data
python experiments/train.py model=ksa_freq data=real detector=external training=spectral_only \
       training.w_mss=0.05 training.w_sot=1.0
# Harmonics-plus-Noise baselines (HpN / HpN+)
python experiments/train.py model=hn      data=real detector=external training=spectral_only
python experiments/train.py model=hn_tcn  data=real detector=external training=spectral_only

# P+Audio is trained on synthetic data (data=synth) for the synthetic experiment and on
# mixed Synth+Real (data=mix) for the real-world experiment.

# Cluster helpers:
sbatch experiments/jobs/run_pilot.sh  ksa_time synth          # <model> <data>
sbatch experiments/jobs/run_ablate.sh ksa_time mix onset      # stop-gradient ablation: <model> <data> <onset|f0|both>
```
Checkpoints are written to `experiments/checkpoints/<tag>_<timestamp>_best.ckpt`.

### 4. Evaluation (Tables 3–4 + audio examples)
With the checkpoints in `experiments/checkpoints/`, run:
```bash
python experiments/evaluate.py --mode all     # -> experiments/evaluation/{synthetic,nsynth}_results.csv
# or a subset: python experiments/evaluate.py --mode nsynth --tags fKSA_E2E_mix,tKSA_E2E_mix --out_csv /tmp/r.csv
# cluster (1h chunks): sbatch experiments/jobs/run_eval_chunk.sh nsynth <tags> <out_csv>
```
This also renders the prediction audio under `experiments/evaluation/audio/`, which the
project page (`index.html`) plays.

### 5. Figures
```bash
figures/nsynth_tKSA_vs_fKSA.ipynb   # Fig. 4 (time-aliasing of the fKSA forward pass)
```

### Pre-trained checkpoints
The checkpoints used to produce the reported numbers ship in `experiments/checkpoints/`
(naming: `{oKSA,fKSA,tKSA,HN,HNtcn}_{E2E,xDet}_{synth,mix,real}[_sgOnset|_sgF0|_sgBoth]_*_best.ckpt`),
so evaluation in step 4 is reproducible without retraining.
