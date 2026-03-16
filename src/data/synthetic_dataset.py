"""Event-based synthetic dataset for Karplus-Strong sound matching.

Each sample consists of a fixed number of pluck *events* on a string
with constant f0.  Between events, all synthesis parameters hold their
values (step-function interpolation).  No slides, vibrato, or pitch
changes — the model is a "high-precision event transcriber".

Output format (per sample)::

    {
        "audio":    [num_audio_samples]            float32
        "events": {
            "exists":         [MAX_EVENTS]         1.0 or 0.0
            "time":           [MAX_EVENTS]         ∈ [0, 1]
            "f0":             [MAX_EVENTS]         Hz
            "burst_gain":     [MAX_EVENTS]         ∈ [0, 1]
            "decay":          [MAX_EVENTS]         ∈ [DECAY_MIN, DECAY_MAX]
            "a1":             [MAX_EVENTS]         ∈ [DAMPING_MIN, DAMPING_MAX]
            "pluck_position": [MAX_EVENTS]         ∈ [PLUCK_POS_MIN, PLUCK_POS_MAX]
            "dynamic_level":  [MAX_EVENTS]         ∈ [DYN_MIN, DYN_MAX]
        }
        "n_events": int
    }
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset
from src.synths.synth import Synth, SynthConfig
from src.synths.param_registry import (
    MIDI_D1, MIDI_D6,
    midi_to_hz,
    PLUCK_POSITION_MIN, PLUCK_POSITION_MAX,
    DAMPING_MIN, DAMPING_MAX,
    DECAY_MIN, DECAY_MAX,
    MAX_EVENTS,
)


class SyntheticDataset(IterableDataset):
    """Event-based KS synthetic dataset.

    Generates random pluck events on a string with constant f0.

    Args:
        num_samples_per_epoch: Items per epoch.
        num_audio_samples:     Audio length in samples.
        fs:                    Sample rate.
        lagrange_order:        Lagrange interpolation order.
        random_seed:           Base random seed.
        max_events_per_sample: Max events to generate.
        min_events_per_sample: Min events per sample.
    """

    def __init__(
        self,
        num_samples_per_epoch: int,
        num_audio_samples: int = 64000,
        fs: int = 16000,
        lagrange_order: int = 5,
        random_seed: int = 42,
        max_events_per_sample: int = 10,
        min_events_per_sample: int = 1,
    ):
        super().__init__()
        self.num_samples_per_epoch = num_samples_per_epoch
        self.num_audio_samples = num_audio_samples
        self.fs = fs
        self.lagrange_order = lagrange_order
        self.random_seed = random_seed
        self.max_events_per_sample = min(max_events_per_sample, MAX_EVENTS)
        self.min_events_per_sample = min_events_per_sample
        self.epoch = 0
        self.duration_s = num_audio_samples / fs

        self.priors = {
            # Event timing
            "min_gap_s":     0.05,
            "first_onset":   dict(lo=0.0, hi=0.3),
            # Mute Probabilities
            "mute_prob": 0.3,
            "mute_decay_reduction": dict(lo=0.05, hi=0.30),
            "mute_a1_increase": dict(lo=0.05, hi=0.30),
            "mute_dyn_reduction": dict(lo=0.20, hi=0.95),
            # Synth params
            "burst_gain":    dict(mean=0.5, conc=3, lo_db=-40.0, hi_db=0.0),
            "pluck_position": dict(mean=0.25, conc=5,
                                   lo=PLUCK_POSITION_MIN, hi=PLUCK_POSITION_MAX),
            "dynamic_level": dict(mean=0.5, conc=3, lo=0.0, hi=1.0),
            "a1":            dict(mean=0.3, conc=5, lo=DAMPING_MIN, hi=DAMPING_MAX),
            "decay":         dict(mean=0.95, conc=5, lo=DECAY_MIN, hi=DECAY_MAX),
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _sample_beta(self, rng, mean: float, conc: float) -> float:
        a = mean * conc
        b = (1 - mean) * conc
        return float(rng.beta(a, b))

    def _mirror(self, val: float, lo: float, hi: float) -> float:
        while val < lo or val > hi:
            if val < lo:
                val = lo + (lo - val)
            elif val > hi:
                val = hi - (val - hi)
        return val

    def _sample_linear(self, rng, prior: dict) -> float:
        """Sample from beta distribution scaled to [lo, hi]."""
        raw = self._sample_beta(rng, prior["mean"], prior["conc"])
        span = prior["hi"] - prior["lo"]
        return prior["lo"] + self._mirror(raw * span, lo=0.0, hi=span)

    def _sample_db(self, rng, prior: dict) -> float:
        """Sample in dB space, return linear amplitude."""
        raw = self._sample_beta(rng, prior["mean"], prior["conc"])
        db = prior["lo_db"] + raw * (prior["hi_db"] - prior["lo_db"])
        return 10.0 ** (db / 20.0)

    def _a1_max_for_midi(self, midi: float) -> float:
        t = (midi - MIDI_D1) / (MIDI_D6 - MIDI_D1)
        return self._mirror(0.8 - t * 0.7, lo=0.1, hi=0.8)

    def _a1_max_for_fs(self) -> float:
        if self.fs <= 16000:  return 0.7
        if self.fs <= 32000:  return 0.8
        if self.fs <= 44100:  return 0.9
        return 1.0

    def _a1_max(self, midi: float) -> float:
        return min(self._a1_max_for_midi(midi), self._a1_max_for_fs())

    def _generate_events(self, rng) -> tuple[dict[str, np.ndarray], int]:
        """Generate a set of pluck and mute events.

        Returns:
            events:   {name: [MAX_EVENTS]} numpy arrays, padded with zeros.
            n_events: actual number of real events.
        """
        p = self.priors
        dur = self.duration_s

        # ── Sample constant global properties for this sample ───────────
        midi = rng.uniform(MIDI_D1, MIDI_D6)
        f0_hz = float(midi_to_hz(midi))
        a1_max = self._a1_max(midi)
        global_burst_gain = self._sample_db(rng, p["burst_gain"])

        # ── Determine number of events ──────────────────────────────────
        n_events = rng.integers(self.min_events_per_sample,
                                self.max_events_per_sample + 1)

        # ── Sample onset times ──────────────────────────────────────────
        max_onset_s = dur * 0.75  # no events in the last quarter
        first_lo = p["first_onset"]["lo"]
        first_hi = p["first_onset"]["hi"]
        first_time = rng.uniform(first_lo, first_hi)

        times = [first_time]
        for _ in range(n_events - 1):
            min_next = times[-1] + p["min_gap_s"]
            if min_next >= max_onset_s:
                break
            next_time = rng.uniform(min_next, max_onset_s)
            times.append(next_time)

        n_events = len(times)
        times_arr = np.array(times, dtype=np.float64) / dur  # normalise to [0, 1]

        # ── Initialise padded event arrays ──────────────────────────────
        events = {
            "exists":         np.zeros(MAX_EVENTS, dtype=np.float32),
            "time":           np.zeros(MAX_EVENTS, dtype=np.float32),
            "f0":             np.zeros(MAX_EVENTS, dtype=np.float32),
            "burst_gain":     np.zeros(MAX_EVENTS, dtype=np.float32),
            "decay":          np.zeros(MAX_EVENTS, dtype=np.float32),
            "a1":             np.zeros(MAX_EVENTS, dtype=np.float32),
            "pluck_position": np.zeros(MAX_EVENTS, dtype=np.float32),
            "dynamic_level":  np.zeros(MAX_EVENTS, dtype=np.float32),
        }

        # ── Fill real events ────────────────────────────────────────────
        prev_was_mute = False  # Track state to prevent consecutive mutes

        for i in range(n_events):
            events["exists"][i] = 1.0
            events["time"][i] = times_arr[i]
            events["f0"][i] = f0_hz

            # Distance/Preamp gain is static for the ENTIRE recording
            events["burst_gain"][i] = global_burst_gain

            is_mute = (i > 0) and (not prev_was_mute) and (rng.random() < p["mute_prob"])

            if is_mute:
                # Mute: Hand hits the string, causing a heavily reduced excitation
                dyn_reduction = rng.uniform(p["mute_dyn_reduction"]["lo"], p["mute_dyn_reduction"]["hi"])
                events["dynamic_level"][i] = events["dynamic_level"][i - 1] * (1.0 - dyn_reduction)

                # Subtract 5% to 30% from the previous decay's energy
                decay_reduction = rng.uniform(p["mute_decay_reduction"]["lo"], p["mute_decay_reduction"]["hi"])
                events["decay"][i] = events["decay"][i - 1] * (1.0 - decay_reduction)

                # Add 5% to 30% to the previous damping's energy
                a1_increase = rng.uniform(p["mute_a1_increase"]["lo"], p["mute_a1_increase"]["hi"])
                events["a1"][i] = events["a1"][i - 1] * (1.0 + a1_increase)

                prev_was_mute = True

            else:
                # Standard Pluck
                events["dynamic_level"][i] = self._sample_linear(rng, p["dynamic_level"])
                events["decay"][i] = self._sample_linear(rng, p["decay"])
                events["a1"][i] = self._sample_linear(
                    rng, {**p["a1"], "hi": min(p["a1"]["hi"], a1_max)}
                )

                prev_was_mute = False

            # Pluck position changes in both cases
            events["pluck_position"][i] = self._sample_linear(rng, p["pluck_position"])

        return events, n_events

    # ── Audio rendering ─────────────────────────────────────────────────

    def _synthesise(self, events: dict[str, np.ndarray], n_events: int) -> np.ndarray:
        params_torch = {
            k: torch.from_numpy(v).unsqueeze(0).float()
            for k, v in events.items()
        }

        config = SynthConfig(
            num_samples=self.num_audio_samples,
            fs=self.fs,
            lagrange_order=self.lagrange_order,
        )

        synth = Synth(config)
        audio, _ = synth.oracle_synth(params_torch)
        return audio.squeeze(0).numpy()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_to_yield = self.num_samples_per_epoch
            worker_id = 0
        else:
            per_worker = self.num_samples_per_epoch // worker_info.num_workers
            leftover = self.num_samples_per_epoch % worker_info.num_workers
            num_to_yield = per_worker + (1 if worker_info.id < leftover else 0)
            worker_id = worker_info.id

        base_seed = self.random_seed + self.epoch * 1000 + worker_id
        rng = np.random.default_rng(base_seed)

        for _ in range(num_to_yield):
            sample_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
            events, n_events = self._generate_events(sample_rng)
            audio = self._synthesise(events, n_events)

            yield {
                "audio": torch.from_numpy(audio).float(),
                "events": {k: torch.from_numpy(v).float()
                           for k, v in events.items()},
                "n_events": n_events,
            }


# ═════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating 4 event-based samples...")
    ds = SyntheticDataset(
        num_samples_per_epoch=4,
        num_audio_samples=64000,
        fs=16000,
        random_seed=123,
    )

    for i, sample in enumerate(ds):
        audio = sample["audio"]
        events = sample["events"]
        n = sample["n_events"]

        print(f"\nSample {i}: {n} events, audio shape {audio.shape}")
        print(f"  audio range: [{audio.min():.4f}, {audio.max():.4f}]")

        for j in range(n):
            print(f"  Event {j}: time={events['time'][j]:.3f} "
                  f"f0={events['f0'][j]:.1f}Hz "
                  f"gain={events['burst_gain'][j]:.3f} "
                  f"decay={events['decay'][j]:.4f} "
                  f"a1={events['a1'][j]:.3f} "
                  f"pos={events['pluck_position'][j]:.3f} "
                  f"dyn={events['dynamic_level'][j]:.3f}")

        # Verify padding
        assert events["exists"][:n].sum() == n
        assert events["exists"][n:].sum() == 0
        assert audio.shape == (64000,)
        assert not torch.isnan(audio).any()

    print("\n✓ All smoke tests passed.")