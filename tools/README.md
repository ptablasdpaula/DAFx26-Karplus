# tools

## `precompute_landscapes.py`

Samples the loss surfaces that the Section 2 demos descend, using the paper's
own synthesiser and losses rather than a browser reimplementation.

The browser cannot run torch, so the surfaces are sampled here and shipped as a
quantised binary. The demos then interpolate that surface and do gradient
descent on it — which means the divergence a reader sees between the
time-domain and frequency-sampling markers is the genuine one, caused by two
different landscapes rather than by anything faked in JavaScript.

### Running it

Needs a checkout of the code repository (the `main` branch of
`ptablasdpaula/DAFx26-Karplus`) with its pixi environment built, because it
imports `src/synths/synth.py` and `src/losses.py` directly:

```bash
git clone -b main https://github.com/ptablasdpaula/DAFx26-Karplus.git ../DAFx26-Karplus-main
cd ../DAFx26-Karplus-main && pixi install && cd -

KARPLUS_REPO=../DAFx26-Karplus-main \
  ../DAFx26-Karplus-main/.pixi/envs/default/bin/python tools/precompute_landscapes.py
```

`KARPLUS_REPO` defaults to a sibling directory named `DAFx26-Karplus-main`, so
if you clone it there the variable can be omitted. Everything runs forward-only on CPU; expect roughly 2.5 hours.

### Output

```text
assets/data/landscapes.bin    uint16-quantised loss grids
assets/data/landscapes.json   axes, per-slice min/max, byte offsets
```

Landscape A is decay x damping, 96 x 96, for tKSA plus fKSA at four FFT sizes,
at onsets of 0 s and 3 s. Landscape B is onset time x f0, 96 x 96, for tKSA and
fKSA, against a 3 x 3 lattice of target positions — the surface is defined
relative to a target, so a freely draggable target is not precomputable and the
demo snaps to that lattice. Landscape B stores L_MSS and L_SOT as separate
grids so the page can reweight them live.

Everything runs at the paper's 16 kHz. This is not negotiable for the f0 story:
sampled at 4 kHz a 110 Hz string keeps only 18 harmonics below Nyquist against
72 here, and the f0 gradient then measures near chance for *both* losses,
contradicting Table 1. At 16 kHz, L_MSS recovers 70-79% on a one-parameter
sweep, in line with the paper.

### Sanity checks

After regenerating, confirm the numbers still say what Section 2 claims:

- every tKSA slice of landscape A should have its minimum within a grid cell of
  the true target, `g = 0.99`, `a1 = 0.2`;
- fKSA slices should be displaced away from it, and markedly flatter — the
  printed `[min, max]` range for a short FFT is much narrower than tKSA's.

If the fKSA slices are *not* displaced, something is wrong: the time-aliasing
this whole section is about has gone missing.
