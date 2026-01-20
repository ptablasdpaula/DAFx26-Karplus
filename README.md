# Designing a Differentiable Karplus-Strong Algorithm for Sound Matching and Timbre Transfer
## Part 1: Differentiable Karplus-Strong
The idea is to design a fully differentiable Karplus-Strong Algorithm w.r.t.
- Excitation Time
- Pluck Position (the length of an all-zero comb-filter)
- Pluck Dynamics (burst gain and burst low-pass filter)
- Fundamental Frequency (Length of delay line)
- Timbre (Coefficient of an in-loop IIR)
- Decay (In-loop Gain)
  
All of these parameters are differentiable through frequency-sampling, except excitation time, which will need of Gumbel-SoftMax. Sound Matching/Timbre Transfer need of a frame-based approach (continuous input-control). This will be used as a proxy for training. Additionally we should compare this implementation with an "oracle" time-domain implementation (no torchLPC, etc..., simple loop synthesis). This will be used for synthetic data generation, for comparison and at inference.

#### Implementation References:
- Frequency-Sampling Library: [FLAMO](https://github.com/gdalsanto/flamo)
- Frame-Based Frequency-Sampling Paper Code: [Gradient-based optimisation of modulation effects](https://github.com/a-carson/modulation_fx)
- Gumbel-Soft Max for Categorical Synths Code: [DiffMoog](https://github.com/aisynth/diffmoog)

## Part 2: Neural Training: Generative Model with Discriminative Losses
The Neural Network consists of:
- Temporal Convolutional Network (TCN)
- Deep Sigmoidal Flows

With Training on these losses:
- Parameter Losses (on synthetic data)
- Spectral Losses
- Discriminative

In order to measure the effect of each loss, we will use the following baselines:
- Supervised, trained only on Parameter Losses
- DDSP, trained only on Spectral Losses
- Semi-Supervised, trained on Parameter Losses and Spectral Losses (both on synthetic and out-of-domain data)
- Discriminative, like the previous but with discriminative losses.

#### Implementation References:
- TCN: [DDX7](https://github.com/fcaspe/ddx7)
- [Deep Sigmoidal Flows](https://github.com/peladeaucome/DAFx_Params_Distrib)
- [Supervised Model paper](https://www.dafx.de/paper-archive/2017/papers/DAFx17_paper_27.pdf):
- [Semi-Supervised Model paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10017350)
- [Discriminative Model paper](https://joshreiss.github.io/documents/2024/Zong2024Machine.pdf)

