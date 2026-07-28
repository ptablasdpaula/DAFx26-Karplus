# Sound Matching with a Differentiable Karplus-Strong Algorithm

Official accompanying repository for the DAFx26 paper.

Audio examples and project page: https://ptablasdpaula.github.io/DAFx26-Karplus/

This project frames plucked-instrument sound matching as event-based parameter
estimation. An encoder predicts discrete pluck events that drive a differentiable
extended Karplus-Strong decoder, using either a time-domain implementation
(`tKSA`) or a frequency-sampling implementation (`fKSA`). The paper compares
three training regimes:

- `p_only`: parameter supervision only
- `p_audio`: parameter supervision plus audio losses
- `audio_only`: audio losses with external onset/f0 detectors

The Harmonics-plus-Noise baselines are named `hpn` and `hpn_p`.

## Repository Layout

```text
src/                      Core model, encoder, decoders, detectors, losses, metrics.
src/synths/               Karplus-Strong DSP/DDSP implementations.
data/                     Synthetic dataset and NSynth download/preprocessing code.
figures/                  Paper figure notebooks.
experiments/              Training, evaluation, gradient analysis, and artifacts.
experiments/checkpoints/  Trained checkpoints used for the paper.
experiments/evaluation/   Result CSVs and audio examples.
experiments/jobs/         Minimal SLURM wrappers for train/eval.
docs/                     GitHub Pages source.
```

## Install

Install Pixi, then create the default CPU environment:

```bash
pixi install
```

For CUDA runs, build the CUDA environment on a GPU node:

```bash
CONDA_OVERRIDE_CUDA=12.9 PIXI_CACHE_DIR=/tmp/pixi-$USER MAX_JOBS=2 pixi install -e cuda
CUDA_HOME="$PWD/.pixi/envs/cuda" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
  .pixi/envs/cuda/bin/python -m pip install --force-reinstall --no-deps \
  --no-build-isolation --no-cache-dir torchlpc==0.7.2
```

Using `/tmp` for `PIXI_CACHE_DIR` avoids incomplete rattler cache extraction on
shared filesystems.

## Reproduce From Checkpoints

The trained checkpoints are committed under `experiments/checkpoints/`. To
recompute Tables 3-4 and render audio examples:

```bash
pixi run -e cuda python experiments/evaluate.py --mode all
```

Outputs:

```text
experiments/evaluation/synthetic_results.csv
experiments/evaluation/real_results.csv
experiments/evaluation/audio/synthetic/
experiments/evaluation/audio/real/
```

The evaluation CLI still uses `--mode nsynth` for the real-guitar dataset loader,
but paper-facing output paths use `real`.

## Data

Synthetic data is generated on the fly by `data/synthetic_dataset.py`.

For NSynth acoustic guitar:

```bash
python data/nsynth/download.py
bash data/nsynth/run_preprocessing.sh
```

## Gradient Analysis

Run the notebook from the `experiments/` working directory so the CSVs are written
next to the notebook:

```bash
pixi run python -m jupyter nbconvert \
  --to notebook --execute experiments/gradient_analysis.ipynb \
  --output /tmp/gradient_analysis.executed.ipynb \
  --ExecutePreprocessor.cwd=experiments \
  --ExecutePreprocessor.timeout=-1
```

Regenerated outputs, ignored by git:

```text
experiments/gradient_accuracy_single.csv
experiments/gradient_accuracy_multi_event.csv
```

## SLURM Jobs

Training and evaluation require a CUDA GPU. The job wrappers are SLURM-only and
do not hardcode any project-specific cluster account or partition. Copy the
example config and fill in your local account, partition, QoS, and GPU request
style:

```bash
cp experiments/jobs/.env.example experiments/jobs/.env
```

The `.env` file is ignored by git, so cluster-specific settings stay local.

## Train All Models

The experiment matrix lives in `experiments/jobs/experiments.conf`. Submit the
full training matrix with:

```bash
experiments/jobs/run_all_training.sh
```

To submit one row from the experiment matrix, pass that row's Hydra overrides to
`run_training.sh`. For example, the real-data P+Audio fKSA run is:

```bash
experiments/jobs/run_training.sh \
  data=mix detector=end_to_end training=combined model=ksa_freq
```

The matrix is organised by the paper-facing objectives:

```text
p_only      parameter-supervised KS oracle baseline
p_audio     KS model trained with parameter supervision and audio losses
audio_only  model trained with audio losses and external onset/f0 detectors
```

## Evaluate All Models

Submit both evaluation jobs with:

```bash
experiments/jobs/run_all_evaluation.sh
```

Direct Python command:

```bash
pixi run -e cuda python experiments/evaluate.py --mode all
```

Evaluate a subset:

```bash
pixi run -e cuda python experiments/evaluate.py --mode nsynth \
  --tags real_fKSA_p_audio,real_tKSA_p_audio \
  --out_csv /tmp/real_subset.csv
```

## Artifact Naming

Checkpoints are named:

```text
{synth,real}_{model}_{objective}[_detach_onset|_detach_f0|_detach_both]_{timestamp}_{best|config}
```

Examples:

```text
synth_oKSA_p_only_20260323_200226_best.ckpt
synth_fKSA_p_audio_detach_f0_20260615_235824_best.ckpt
real_tKSA_p_audio_20260615_235824_best.ckpt
real_hpn_p_audio_only_20260618_003348_best.ckpt
```

Audio examples use the same paper-facing objective names:

```text
experiments/evaluation/audio/synthetic/pred/fksa/p_audio/detach_onset/
experiments/evaluation/audio/real/pred/hpn_p/audio_only/
```

## Job Scripts

Only the minimal training/evaluation wrappers are kept:

```text
experiments/jobs/run_all.sh
experiments/jobs/run_all_training.sh
experiments/jobs/run_all_evaluation.sh
experiments/jobs/run_training.sh
experiments/jobs/run_evaluation.sh
```
