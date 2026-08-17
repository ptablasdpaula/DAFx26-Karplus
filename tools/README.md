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
assets/data/landscapes.bin    uint16-quantised grids
assets/data/landscapes.json   axes, per-slice min/max, byte offsets
```

Three groups are stored. `A` and `B` are loss surfaces. `G` is something
different and easy to confuse with them: the **true autograd gradients** of the
loss with respect to onset time and f0, on a 48 x 48 grid.

That distinction carries Section 2.3. Onset reaches the synthesiser through a
Straight-Through Estimator, so the loss surface has usable slope toward the
right onset — finite-differencing it scores 69.4% on the sign test — while the
gradient the model actually receives scores 54.4%, matching Table 1's 51.2%.
Demo B descends `G`, not the surface; descending the surface would show the
markers confidently finding the correct onset, i.e. the opposite of the result.
The field is looked up by nearest cell and never interpolated, because
smoothing it would manufacture the directional coherence it exists to show is
missing.

To recompute only the gradient field and append it to existing output:

```bash
PRECOMPUTE_ONLY=G KARPLUS_REPO=../DAFx26-Karplus-main \
  ../DAFx26-Karplus-main/.pixi/envs/default/bin/python tools/precompute_landscapes.py
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
  printed `[min, max]` range for a short FFT is much narrower than tKSA's;
- the onset sign accuracy of `G` should sit near 51%, and clearly below the
  same measurement taken off the loss surface.

`tests/data-check.html` asserts all of this in the browser:

```bash
python3 -m http.server 8000 &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --virtual-time-budget=180000 --dump-dom http://localhost:8000/tests/data-check.html
```

If the fKSA slices are *not* displaced, something is wrong: the time-aliasing
this whole section is about has gone missing.
