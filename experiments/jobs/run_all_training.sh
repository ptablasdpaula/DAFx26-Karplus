#!/bin/bash
# submit_all_jobs.sh
# Run this from the root DAFx26-Karplus directory: ./submit_all_jobs.sh

set -e

JOB_SCRIPT="experiments/jobs/run_training.sh"

echo "🚀 Submitting all Sound Matching Training Experiments to SLURM..."

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC EVAL BASELINES (Synthetic Only Data)
# ═════════════════════════════════════════════════════════════════════════════

sbatch $JOB_SCRIPT experiment=synth_eval data=synthetic_only detector=none training=param_only model=ks_freqsampling
sbatch $JOB_SCRIPT experiment=synth_eval data=synthetic_only detector=none training=param_only model=ks_timedomain

sbatch $JOB_SCRIPT experiment=synth_eval data=synthetic_only detector=none training=combined model=ks_freqsampling
sbatch $JOB_SCRIPT experiment=synth_eval data=synthetic_only detector=none training=combined model=ks_timedomain

sbatch $JOB_SCRIPT experiment=synth_eval data=synthetic_only detector=external training=spectral_only model=ks_freqsampling
sbatch $JOB_SCRIPT experiment=synth_eval data=synthetic_only detector=external training=spectral_only model=ks_timedomain

# ═════════════════════════════════════════════════════════════════════════════
# 2. NSYNTH OOD BASELINES
# ═════════════════════════════════════════════════════════════════════════════

# param_only OOD jobs removed: no spectral loss means val_ood/loss is
# meaningless and early stopping kills them prematurely. Instead, evaluate.py
# tests the Synth_Free_*_Super checkpoints (trained in section 1) on NSynth.

sbatch $JOB_SCRIPT experiment=ood_eval data=synthetic_and_nsynth detector=none training=combined model=ks_freqsampling
sbatch $JOB_SCRIPT experiment=ood_eval data=synthetic_and_nsynth detector=none training=combined model=ks_timedomain

sbatch $JOB_SCRIPT experiment=ood_eval data=nsynth_only detector=external training=spectral_only model=ks_freqsampling
sbatch $JOB_SCRIPT experiment=ood_eval data=nsynth_only detector=external training=spectral_only model=ks_timedomain

# ── The Ultimate Baseline: DDSP Harmonics + Noise ──
sbatch $JOB_SCRIPT experiment=ood_eval data=nsynth_only detector=external training=spectral_only model=harmonics_noise

echo "✅ All 11 jobs successfully submitted to SLURM!"