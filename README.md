# Checkpoints

Trained weights for every run reported in the DAFx26 paper. They live on this
orphan branch rather than on `main` so that cloning the code does not pull
~280 MB of Git LFS objects along with it.

Code, and everything else: [`main`](../../tree/main).

## Fetching

From a checkout of `main`:

```bash
scripts/fetch_checkpoints.sh
```

That drops the files into `experiments/checkpoints/`, where `evaluate.py` looks
for them by default. It needs `git-lfs` installed (`git lfs install`), otherwise
you get 132-byte pointer files instead of weights.

Manually, if you prefer:

```bash
git fetch origin checkpoints
git checkout origin/checkpoints -- experiments/checkpoints
git restore --staged experiments/checkpoints
```

## Naming

```text
{synth,real}_{model}_{objective}[_detach_onset|_detach_f0|_detach_both]_{timestamp}_{best|config}
```

| Field       | Values                                                          |
|-------------|-----------------------------------------------------------------|
| domain      | `synth` (synthetic training data), `real` (NSynth guitar)        |
| model       | `tKSA`, `fKSA`, `oKSA`, `hpn`, `hpn_p`                           |
| objective   | `p_only`, `p_audio`, `audio_only`                                |
| detach      | stop-gradient ablation on the audio loss path, where present     |
| suffix      | `_best.ckpt` (weights), `_config.yaml` (the resolved Hydra config)|

Examples:

```text
synth_oKSA_p_only_20260323_200226_best.ckpt
synth_fKSA_p_audio_detach_f0_20260615_235824_best.ckpt
real_tKSA_p_audio_20260615_235824_best.ckpt
real_hpn_p_audio_only_20260618_003348_best.ckpt
```

Each `_best.ckpt` is paired with the `_config.yaml` of the same tag and
timestamp. `evaluate.py` requires both: it resolves a tag to its newest matching
timestamp and rebuilds the model from the stored config, so an orphaned
checkpoint with no config is skipped.
