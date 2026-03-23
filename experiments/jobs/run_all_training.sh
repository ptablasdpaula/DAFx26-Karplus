#!/bin/bash
# submit_all_jobs.sh
# Run this from the root DAFx26-Karplus directory: ./submit_all_jobs.sh

set -e

JOB_SCRIPT="experiments/jobs/run_training.sh"

echo "🚀 Submitting all Sound Matching Training Experiments to SLURM..."

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC EVAL BASELINES (Synthetic Only Data)
# ═════════════════════════════════════════════════════════════════════════════

sbatch $JOB_SCRIPT data=synth detector=end_to_end training=param_only model=ksa_oracle

sbatch $JOB_SCRIPT data=synth detector=end_to_end training=combined model=ksa_freq
sbatch $JOB_SCRIPT data=synth detector=end_to_end training=combined model=ksa_time

sbatch $JOB_SCRIPT data=synth detector=external training=spectral_only model=ksa_freq
sbatch $JOB_SCRIPT data=synth detector=external training=spectral_only model=ksa_time

# ═════════════════════════════════════════════════════════════════════════════
# 2. NSYNTH OOD BASELINES
# ═════════════════════════════════════════════════════════════════════════════

sbatch $JOB_SCRIPT data=mix detector=end_to_end training=combined model=ksa_freq
sbatch $JOB_SCRIPT data=mix detector=end_to_end training=combined model=ksa_time

sbatch $JOB_SCRIPT data=real detector=external training=spectral_only model=ksa_freq
sbatch $JOB_SCRIPT data=real detector=external training=spectral_only model=ksa_time

# ── The Ultimate Baseline: DDSP Harmonics + Noise ──
sbatch $JOB_SCRIPT data=real detector=external training=spectral_only model=hn
sbatch $JOB_SCRIPT data=real detector=external training=spectral_only model=hn_tcn

echo "✅ All 11 jobs successfully submitted to SLURM!"