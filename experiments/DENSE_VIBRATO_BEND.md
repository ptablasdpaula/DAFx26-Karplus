# Dense vibrato--bend inverse experiment

This experiment directly fits 250-frame Karplus--Strong controls to one
matched-renderer four-second A2 gesture. It is an optimization-closure study,
not an encoder experiment. Outputs are written below the ignored
`experiments/outputs/dense-vibrato-bend-ot/` directory.

The three production objectives are normalized by their initial rendered-audio
gradient RMS for each seed:

- `mss`: normalized MSS plus the frame-normalized effective gate count;
- `ot`: normalized canonical-geometry 2D-OT plus the gate term;
- `hybrid`: normalized 2D-OT plus `0.01` normalized MSS plus the gate term.

Useful commands:

```bash
# A deliberately short CPU-only closure check.
pixi run python experiments/dense_vibrato_bend.py smoke

# One complete GPU run.
pixi run -e cuda python experiments/dense_vibrato_bend.py run \
  --arm hybrid --lambda-sparse 0.01 --seed 2027

# All 27 fits in the current process, resuming existing checkpoints.
pixi run -e cuda python experiments/dense_vibrato_bend.py sweep

# Exact resumption of one checkpoint.
pixi run -e cuda python experiments/dense_vibrato_bend.py resume \
  --run-dir experiments/outputs/dense-vibrato-bend-ot/arm=hybrid__lambda=0.01__seed=2027

# Rebuild tables and Pareto plots from completed result JSON files.
pixi run python experiments/dense_vibrato_bend.py aggregate
```

On Andrena, use `experiments/jobs/submit_dense_vibrato_bend.sh`. It first
submits an A100 CUDA-extension preflight, then a nine-element array (one
arm/lambda pair per task, with three seeds sequentially), then dependent
aggregation. Production commands refuse CUDA execution unless the active GPU
is `sm_80` and `torchlpc.EXTENSION_LOADED` is true.
