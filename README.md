# Sound Matching with a Differentiable Karplus-Strong Algorithm
This is the official accompanying repositorty for the DAFx26 submission.

The repo is organised as such:
```
figures/         <- Notebooks with all figures shown in paper.
src/             <- Where decoder, encoder, model and detectors live
src/data         <- Data filtering, synthetic data generation
src/synths       <- Karplus-Strong DSP and DDSP implementations
experiments/     <- Everything related to the gradient analysis and sound matching experiments
```

## Installation
1. Install Pixi
```bash
brew install pixi
```

2. Clone the repo
```bash
git clone https://github.com/ptablasdpaula/DAFx26-Karplus.git
cd DAFx26-Karplus
```

3. Install Dependencies
```bash
pixi install
```

4. Activate Environment
```bash
pixi shell
```
