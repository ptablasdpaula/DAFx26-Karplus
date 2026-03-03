import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import IterableDataset
from src.synths.synth import Synth, SynthConfig

NOTE_FREQS = {'E1': 41.20, 'B5': 987.77}
F0_MIN_HZ = NOTE_FREQS['E1']
F0_MAX_HZ = NOTE_FREQS['B5']
MIDI_E1 = 28
MIDI_B5 = 83

def midi_to_hz(midi):
    return 440 * 2 ** ((midi - 69) / 12.0)

def hz_to_midi(hz):
    return 69 + 12 * np.log2(hz / 440)


@dataclass
class Segments:
    """
    Compact representation of a note segment sequence.

    Attributes:
        indices    : (N,) int array   — frame index of each segment boundary
        f0_hz      : (N,) float array — fundamental frequency in Hz per segment
        is_pluck   : (N,) bool array  — whether a pluck fires at this boundary
        num_frames : total number of frames in the sequence
    """
    indices    : np.ndarray  # (N,) int
    f0_hz      : np.ndarray  # (N,) float
    is_pluck   : np.ndarray  # (N,) bool
    num_frames : int

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return zip(self.indices, self.f0_hz, self.is_pluck)

    def segment_range(self, i: int) -> tuple[int, int]:
        """Return (start, end) frame indices for segment i."""
        start = self.indices[i]
        end   = self.indices[i + 1] if i + 1 < len(self) else self.num_frames
        return start, end


class SyntheticDataset(IterableDataset):
    """
    Karplus-Strong synthetic dataset.

    Args:
        num_samples_per_epoch: How many items per "epoch"
        num_audio_samples: Audio length in samples (default 64000 = 4s @ 16kHz)
        num_frames: Number of control-rate frames
        fs: Sample rate
        lagrange_order: Lagrange interpolation order
        lti: If True, generate LTI (single pluck, static params)
        blend_lti: If True and lti is False, LTI samples are randomly mixed along time-varying samples
        random_seed: Base random seed
    """

    def __init__(
            self,
            num_samples_per_epoch: int,
            num_audio_samples: int = 64000,
            num_frames: int = 250,
            fs: int = 16000,
            lagrange_order: int = 5,
            lti: bool = False,
            blend_lti: bool = True,
            random_seed: int = 42,
    ):
        super().__init__()
        self.num_samples_per_epoch = num_samples_per_epoch
        self.num_audio_samples = num_audio_samples
        self.num_frames = num_frames
        self.fs = fs
        self.lagrange_order = lagrange_order
        self.lti = lti
        self.blend_lti = blend_lti
        self.random_seed = random_seed

        # log_scale=True → sampled in dB space.
        # low/high → linear range for _sample_param.
        self.priors = {
            'first_onset':       dict(mean=0.5,  conc=1,  low_s=0.0,  high_s=0.5),
            'trigger_gap':       dict(mean=0.25, conc=3,  low_s=0.1,  high_s=4.0),
            'prob_note_change':  dict(prob=0.30),
            'prob_octave_shift': dict(prob=0.20),
            'prob_slide':        dict(prob=0.20),
            'prob_vibrato':      dict(prob=0.20),
            'prob_skip_trigger': dict(prob=0.20),
            'vibrato_rate':      dict(low=0.1,   high=7.0),
            'vibrato_depth':     dict(low=0.1,   high=0.5),
            'pluck_position':    dict(mean=0.25,  conc=5,  low=0.01,   high=0.5),
            'burst_gain':        dict(mean=0.5,  conc=5,  low_db=-40.0, high_db=0.0, log_scale=True),
            'dynamic_level':     dict(mean=0.5,  conc=5,  low_db=-40.0, high_db=0.0, log_scale=True),
            'a1':                dict(mean=0.3,  conc=5,  low=0.0,    high=0.75),
            'decay':             dict(mean=0.95, conc=5,  low=0.9,    high=1.0),
        }

        self.ltv_extras = {
            'pluck_position': dict(change_prob=0.70),
            'dynamic_level':  dict(change_prob=0.70),
            'a1':             dict(change_prob=0.70),
            'decay':          dict(change_prob=0.70),
        }

    @property
    def _fps(self) -> float:
        """Frames per second."""
        return self.num_frames * self.fs / self.num_audio_samples

    def _seconds_to_frames(self, s: float) -> int:
        return int(round(s * self._fps))

    def _get_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.random_seed + (worker_info.id if worker_info else 0)
        return np.random.default_rng(seed)

    def _sample_beta(self, rng, mean, concentration, size=1):
        a = mean * concentration
        b = (1 - mean) * concentration
        return rng.beta(a, b, size=size)

    def _mirror(self, val: float, low: float = None, high: float = None) -> float:
        if low is not None and val < low:
            val = low + (low - val)
        if high is not None and val > high:
            val = high - (val - high)
        return val

    def _sample_param(self, rng, prior: dict, high_override: float = None) -> float:
        if prior.get('log_scale'):
            raw = float(self._sample_beta(rng, prior['mean'], prior['conc'])[0])
            db = prior['low_db'] + raw * (prior['high_db'] - prior['low_db'])
            return 10.0 ** (db / 20.0)
        high = high_override if high_override is not None else prior['high']
        low = prior['low']
        raw = float(self._sample_beta(rng, prior['mean'], prior['conc'])[0])
        span = high - low
        return low + self._mirror(raw * span, low=0.0, high=span)

    def _a1_max_for_fs(self) -> float:
        if self.fs <= 16000:  return 0.7
        if self.fs <= 32000:  return 0.8
        if self.fs <= 44100:  return 0.9
        return 1.0

    def _a1_max_for_midi(self, midi: float) -> float:
        t = (midi - MIDI_E1) / (MIDI_B5 - MIDI_E1)
        return self._mirror(0.8 - t * 0.7, low=0.1, high=0.8)

    def _a1_max(self, midi: float) -> float:
        return min(self._a1_max_for_midi(midi), self._a1_max_for_fs())

    def _get_first_onset_frame(self, rng) -> int:
        if self.lti:
            return 0
        p = self.priors['first_onset']
        low_frame  = int(p['low_s']  * self._fps)
        high_frame = int(p['high_s'] * self._fps)
        raw = float(self._sample_beta(rng, p['mean'], p['conc'])[0])
        onset_frame = int(low_frame + raw * (high_frame - low_frame))
        return int(self._mirror(onset_frame, low=low_frame, high=high_frame))

    def _choose_next_midi(self, rng, current_midi: float) -> float:
        ALL_INTERVALS = np.array([-7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7])
        p_note   = self.priors['prob_note_change']['prob']
        p_octave = self.priors['prob_octave_shift']['prob']

        if rng.random() < p_note:
            current_midi += rng.choice(ALL_INTERVALS)

        while rng.random() < p_octave:
            current_midi += rng.choice([12, -12])

        return float(self._mirror(current_midi, low=MIDI_E1, high=MIDI_B5))

    def _f0_period_to_frames(self, f0_hz: float) -> int:
        return max(int(np.ceil(self._fps / f0_hz)), 1)

    def _make_segments(self, rng) -> Segments:
        """
        Sample segment boundary positions, f0 values, and pluck flags.
        Segment generation continues until no further segment fits within num_frames.
        """
        tg             = self.priors['trigger_gap']
        p_skip_trigger = self.priors['prob_skip_trigger']['prob']

        current_midi = rng.uniform(MIDI_E1, MIDI_B5 + 1)
        first_onset  = self._get_first_onset_frame(rng)
        indices      = [first_onset]
        f0_hz        = [midi_to_hz(current_midi)]
        is_pluck     = [True]  # first segment always plucks

        while True:
            last_idx = indices[-1]
            min_gap  = self._f0_period_to_frames(f0_hz[-1])

            if last_idx + min_gap >= self.num_frames:
                break

            raw        = float(self._sample_beta(rng, tg['mean'], tg['conc'])[0])
            gap_s      = tg['low_s'] + raw * (tg['high_s'] - tg['low_s'])
            gap_frames = self._seconds_to_frames(gap_s)
            next_idx   = last_idx + max(gap_frames, min_gap)

            if next_idx >= self.num_frames:
                break

            current_midi = self._choose_next_midi(rng, current_midi)
            indices.append(next_idx)
            f0_hz.append(midi_to_hz(current_midi))
            is_pluck.append(rng.random() >= p_skip_trigger)

        return Segments(
            indices    = np.array(indices,  dtype=int),
            f0_hz      = np.array(f0_hz,    dtype=float),
            is_pluck   = np.array(is_pluck, dtype=bool),
            num_frames = self.num_frames,
        )

    def _apply_slide(self, f0_frames: np.ndarray, segs: Segments, i: int) -> None:
        start, end = segs.segment_range(i)
        f0_frames[start:end] = np.linspace(segs.f0_hz[i], segs.f0_hz[i + 1], end - start)

    def _apply_vibrato(self, rng, f0_frames: np.ndarray, segs: Segments, i: int) -> None:
        start, end    = segs.segment_range(i)
        seg_len       = end - start
        vibrato_rate  = rng.uniform(self.priors['vibrato_rate']['low'], self.priors['vibrato_rate']['high'])
        vibrato_depth = rng.uniform(self.priors['vibrato_depth']['low'], self.priors['vibrato_depth']['high'])
        t             = np.linspace(0, seg_len / self._fps, seg_len)
        vibrato       = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        seg_midi      = hz_to_midi(f0_frames[start:end]) + vibrato
        f0_frames[start:end] = midi_to_hz(np.clip(seg_midi, MIDI_E1, MIDI_B5))

    def _build_f0_trajectory(self, rng, segs: Segments) -> np.ndarray:
        p_slide = self.priors['prob_slide']['prob']
        p_vib   = self.priors['prob_vibrato']['prob']

        f0_frames = np.full(self.num_frames, segs.f0_hz[0], dtype=np.float64)

        for i in range(len(segs)):
            start, end = segs.segment_range(i)
            f0_frames[start:end] = segs.f0_hz[i]

            chance = rng.random()
            if chance < p_slide and i + 1 < len(segs):
                self._apply_slide(f0_frames, segs, i)
            elif chance > 1 - p_vib:
                self._apply_vibrato(rng, f0_frames, segs, i)

        return f0_frames.astype(np.float32)

    def _gen_triggers_and_f0(self, rng) -> tuple[np.ndarray, Segments]:
        segs      = self._make_segments(rng)
        f0_frames = self._build_f0_trajectory(rng, segs)
        return f0_frames, segs

    def _generate_varying_param(self, rng, segment_indices, mean, conc, change_prob,
                                 low=None, high=None, log_scale=False,
                                 low_db=-60.0, high_db=0.0, high_schedule=None):
        """
        Generate a frame-rate parameter that may change at segment boundaries.
        log_scale=True → dB-space sampling; low/high ignored, use low_db/high_db.
        high_schedule   → per-segment upper bounds (e.g. pitch-dependent a1 ceilings).
        """
        num_frames = self.num_frames
        param = np.zeros(num_frames, dtype=np.float32)

        def _draw(high_i):
            raw = float(self._sample_beta(rng, mean, conc)[0])
            if log_scale:
                db = low_db + raw * (high_db - low_db)
                return 10.0 ** (db / 20.0)
            span = high_i - low
            return low + self._mirror(raw * span, low=0.0, high=span)

        current_val = _draw(high_schedule[0] if high_schedule else high)

        for i, start in enumerate(segment_indices):
            end    = segment_indices[i + 1] if i + 1 < len(segment_indices) else num_frames
            high_i = high_schedule[i] if high_schedule else high

            if i > 0 and rng.random() < change_prob:
                current_val = _draw(high_i)
            elif not log_scale:
                span = high_i - low
                current_val = low + self._mirror(current_val - low, low=0.0, high=span)
            param[start:end] = current_val

        if segment_indices[0] > 0:
            param[:segment_indices[0]] = param[segment_indices[0]]

        return param

    def _generate_lti_params(self, rng) -> dict:
        """Single pluck, static parameters throughout."""
        num_frames  = self.num_frames
        onset_frame = self._get_first_onset_frame(rng)

        base_midi = rng.uniform(MIDI_E1, MIDI_B5)
        f0_val    = midi_to_hz(base_midi)
        a1_max    = self._a1_max(hz_to_midi(f0_val))

        p = self.priors

        burst_gain = np.zeros(num_frames, dtype=np.float32)
        burst_gain[onset_frame] = self._sample_param(rng, p['burst_gain'])

        return {
            'f0':             np.full(num_frames, f0_val, dtype=np.float32),
            'burst_gain':     burst_gain,
            'pluck_position': np.full(num_frames, self._sample_param(rng, p['pluck_position']), dtype=np.float32),
            'dynamic_level':  np.full(num_frames, self._sample_param(rng, p['dynamic_level']), dtype=np.float32),
            'a1':             np.full(num_frames, self._sample_param(rng, p['a1'], high_override=a1_max), dtype=np.float32),
            'decay':          np.full(num_frames, self._sample_param(rng, p['decay']), dtype=np.float32),
        }

    def _generate_params(self, rng) -> dict:
        if self.lti or (self.blend_lti and rng.random() < 0.25):
            return self._generate_lti_params(rng)

        f0, segs = self._gen_triggers_and_f0(rng)

        p  = self.priors
        ex = self.ltv_extras

        burst_gain = np.zeros(self.num_frames, dtype=np.float32)
        for idx, hz, pluck in segs:
            if pluck:
                burst_gain[idx] = self._sample_param(rng, p['burst_gain'])

        a1_highs = [self._a1_max(hz_to_midi(hz)) for hz in segs.f0_hz]

        return {
            'f0':             f0,
            'burst_gain':     burst_gain,
            'pluck_position': self._generate_varying_param(rng, segs.indices, **p['pluck_position'], **ex['pluck_position']),
            'dynamic_level':  self._generate_varying_param(rng, segs.indices, **p['dynamic_level'],  **ex['dynamic_level'],  low=0.0, high=1.0),
            'a1':             self._generate_varying_param(rng, segs.indices, **p['a1'],             **ex['a1'],             high_schedule=a1_highs),
            'decay':          self._generate_varying_param(rng, segs.indices, **p['decay'],          **ex['decay']),
        }

    def _synthesise(self, params_np: dict) -> np.ndarray:
        config = SynthConfig(
            num_samples=self.num_audio_samples,
            fs=self.fs,
            lagrange_order=self.lagrange_order,
        )
        synth = Synth(config)
        params_torch = {k: torch.from_numpy(v).unsqueeze(0).float() for k, v in params_np.items()}
        with torch.no_grad():
            audio = synth.oracle_synth(params_torch)  # [1, num_samples]
        return audio.squeeze(0).numpy()

    def __iter__(self):
        rng = self._get_rng()
        yielded = 0
        while yielded < self.num_samples_per_epoch:
            sample_rng = np.random.default_rng(rng.integers(0, 2**31))
            params = self._generate_params(sample_rng)
            audio  = self._synthesise(params)
            yielded += 1
            yield {
                'audio':  torch.from_numpy(audio).float(),
                'params': {k: torch.from_numpy(v).float() for k, v in params.items()},
            }