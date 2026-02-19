import torch
import numpy as np
from torch.utils.data import IterableDataset
from synths.synth import Synth, SynthConfig

NOTE_FREQS = {'E1': 41.20, 'B5': 987.77}
F0_MIN_HZ = NOTE_FREQS['E1']
F0_MAX_HZ = NOTE_FREQS['B5']
MIDI_E1 = 28
MIDI_B5 = 83

def midi_to_hz(midi):
    return 440 * 2 ** ((midi - 69) / 12.0)

def hz_to_midi(hz):
    return 69 + 12 * np.log2(hz / 440)


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

        # Single source of truth for all distributions.
        # log_scale=True → sampled in dB space via _sample_db.
        # low/high → linear range for _sample_param.
        self.priors = {
            'first_onset':       dict(mean=0.5,  conc=1,  low_s=0.0,  high_s=0.5),
            'num_triggers':      dict(mean=0.2,  conc=8,  low_n=2,    high_n=20),
            'trigger_gap':       dict(mean=0.25, conc=3,  low_s=0.1,  high_s=4.0),
            'prob_note_change':  dict(prob=0.30),
            'prob_octave_shift': dict(prob=0.20),
            'prob_slide':        dict(prob=0.20),
            'prob_vibrato':      dict(prob=0.20),
            'prob_skip_trigger': dict(prob=0.20),
            'vibrato_rate':      dict(low=0.1,   high=7.0),
            'vibrato_depth':     dict(low=0.1,   high=0.5),
            'pluck_position':    dict(mean=0.5,  conc=5,  low=0.01,   high=0.99),
            'burst_gain':        dict(mean=0.5,  conc=5,  low_db=-40.0, high_db=0.0, log_scale=True),
            'dynamic_level':     dict(mean=0.5,  conc=5,  low_db=-40.0, high_db=0.0, log_scale=True),
            'a1':                dict(mean=0.3,  conc=5,  low=0.0,    high=0.75),
            'decay':             dict(mean=0.95, conc=5,  low=0.9,    high=1.0),
        }

        self.ltv_extras = {
            'pluck_position': dict(change_prob=0.70),
            'burst_gain':     dict(change_prob=0.70),
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

    def _gen_triggers_and_f0(self, rng, num_triggers_max: int):
        """
        Generate trigger positions and f0 trajectory.
        Each trigger enforces a minimum gap of one delay-line period (L = fs/f0).

        Returns:
            f0_frames : (num_frames,) float32 array
            segments  : (num_segments, 2) int array where each row is
                        [frame_index, is_pluck] — is_pluck is 1 if the
                        segment boundary fires a pluck, 0 if it is silent
                        (note change still occurs, onset is suppressed).
        """
        num_frames = self.num_frames
        ALL_INTERVALS = np.array([-7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7])
        tg    = self.priors['trigger_gap']
        p_note   = self.priors['prob_note_change']['prob']
        p_octave = self.priors['prob_octave_shift']['prob']
        p_slide  = self.priors['prob_slide']['prob']
        p_vib    = self.priors['prob_vibrato']['prob']
        p_skip_trigger = self.priors['prob_skip_trigger']['prob']
        vib_rate_low,  vib_rate_high  = self.priors['vibrato_rate']['low'],  self.priors['vibrato_rate']['high']
        vib_depth_low, vib_depth_high = self.priors['vibrato_depth']['low'], self.priors['vibrato_depth']['high']

        base_midi     = rng.uniform(MIDI_E1, MIDI_B5 + 1)
        first_onset   = self._get_first_onset_frame(rng)
        # segments: list of [frame_index, is_pluck]; first segment always plucks
        segments_list = [[first_onset, 1]]
        trigger_midis = [base_midi]
        current_midi  = base_midi

        for _ in range(num_triggers_max - 1):
            current_hz = midi_to_hz(current_midi)
            min_gap    = max(int(np.ceil(num_frames * self.fs / (self.num_audio_samples * current_hz))), 1)

            if segments_list[-1][0] + min_gap >= num_frames:
                break

            raw        = float(self._sample_beta(rng, tg['mean'], tg['conc'])[0])
            gap_s      = tg['low_s'] + raw * (tg['high_s'] - tg['low_s'])
            gap_frames = self._seconds_to_frames(gap_s)
            next_idx   = segments_list[-1][0] + max(gap_frames, min_gap)
            if next_idx >= num_frames:
                break

            if rng.random() < p_note:
                valid_intervals = ALL_INTERVALS[
                    (current_midi + ALL_INTERVALS >= MIDI_E1) &
                    (current_midi + ALL_INTERVALS <= MIDI_B5)
                ]
                if len(valid_intervals) == 0:
                    valid_intervals = np.array([0])
                interval = rng.choice(valid_intervals)

                cumulative_octave_shift = 0
                while rng.random() < p_octave:
                    candidate_midi = current_midi + interval + cumulative_octave_shift
                    can_go_up   = (candidate_midi + 12) <= MIDI_B5
                    can_go_down = (candidate_midi - 12) >= MIDI_E1
                    if not can_go_up and not can_go_down:
                        break
                    octave_choices = (
                        ([12] if can_go_up else []) +
                        ([-12] if can_go_down else [])
                    )
                    cumulative_octave_shift += rng.choice(octave_choices)
                    if np.abs(cumulative_octave_shift) >= 36:
                        break

                interval += cumulative_octave_shift
                new_midi = float(np.clip(current_midi + interval, MIDI_E1, MIDI_B5))
            else:
                new_midi = current_midi

            trigger_midis.append(new_midi)
            current_midi = new_midi

            is_pluck = 0 if rng.random() < p_skip_trigger else 1
            segments_list.append([next_idx, is_pluck])

        trigger_midis = np.array(trigger_midis)
        segments      = np.array(segments_list, dtype=int)  # (num_segments, 2)
        num_segments  = len(segments)

        f0_frames = np.full(num_frames, midi_to_hz(trigger_midis[0]), dtype=np.float64)

        for i in range(num_segments):
            start  = segments[i, 0]
            end    = segments[i + 1, 0] if i + 1 < num_segments else num_frames
            seg_hz = midi_to_hz(trigger_midis[i])
            f0_frames[start:end] = seg_hz

            do_slide   = (i + 1 < num_segments) and (rng.random() < p_slide)
            do_vibrato = rng.random() < p_vib

            if do_slide and do_vibrato:
                do_slide, do_vibrato = (True, False) if rng.random() < 0.5 else (False, True)

            if do_slide:
                f0_frames[start:end] = np.linspace(seg_hz, midi_to_hz(trigger_midis[i + 1]), end - start)

            if do_vibrato:
                seg_len       = end - start
                vibrato_rate  = rng.uniform(vib_rate_low,  vib_rate_high)
                vibrato_depth = rng.uniform(vib_depth_low, vib_depth_high)
                t = np.linspace(0, seg_len / self._fps, seg_len)
                vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
                seg_midi = hz_to_midi(f0_frames[start:end]) + vibrato
                f0_frames[start:end] = midi_to_hz(np.clip(seg_midi, MIDI_E1, MIDI_B5))

        return f0_frames.astype(np.float32), segments

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

        return param

    def _generate_lti_params(self, rng) -> dict:
        """Single pluck, static parameters throughout."""
        num_frames = self.num_frames
        onset_probs = np.zeros(num_frames, dtype=np.float32)
        onset_probs[self._get_first_onset_frame(rng)] = 1.0

        base_midi = rng.uniform(MIDI_E1, MIDI_B5)
        f0_val    = midi_to_hz(base_midi)
        a1_max    = self._a1_max(hz_to_midi(f0_val))

        p = self.priors
        scalar_params = {
            'pluck_position': self._sample_param(rng, p['pluck_position']),
            'burst_gain':     self._sample_param(rng, p['burst_gain']),
            'dynamic_level':  self._sample_param(rng, p['dynamic_level']),
            'a1':             self._sample_param(rng, p['a1'], high_override=a1_max),
            'decay':          self._sample_param(rng, p['decay']),
        }

        return {
            'onset_probs': onset_probs,
            'f0':          np.full(num_frames, f0_val, dtype=np.float32),
            **{k: np.full(num_frames, v, dtype=np.float32) for k, v in scalar_params.items()},
        }

    def _generate_params(self, rng) -> dict:
        if self.lti or (self.blend_lti and rng.random() < 0.25):
            return self._generate_lti_params(rng)

        nt = self.priors['num_triggers']
        raw = self._sample_beta(rng, nt['mean'], nt['conc'])[0]
        num_triggers_max = int(np.clip(
            np.round(raw * nt['high_n']), nt['low_n'], nt['high_n']
        ))

        f0, segments = self._gen_triggers_and_f0(rng, num_triggers_max)
        # segments[:, 0] → frame indices; segments[:, 1] → is_pluck flags

        onset_probs = np.zeros(self.num_frames, dtype=np.float32)
        for idx, is_pluck in segments:
            if is_pluck:
                onset_probs[idx] = 1.0

        segment_indices = list(segments[:, 0])

        p  = self.priors
        ex = self.ltv_extras
        a1_highs = [self._a1_max(hz_to_midi(float(f0[idx]))) for idx in segment_indices]

        return {
            'onset_probs':    onset_probs,
            'f0':             f0,
            'pluck_position': self._generate_varying_param(rng, segment_indices, **p['pluck_position'], **ex['pluck_position']),
            'burst_gain':     self._generate_varying_param(rng, segment_indices, **p['burst_gain'],     **ex['burst_gain'],     low=0.0, high=1.0),
            'dynamic_level':  self._generate_varying_param(rng, segment_indices, **p['dynamic_level'],  **ex['dynamic_level'],  low=0.0, high=1.0),
            'a1':             self._generate_varying_param(rng, segment_indices, **p['a1'],             **ex['a1'],             high_schedule=a1_highs),
            'decay':          self._generate_varying_param(rng, segment_indices, **p['decay'],          **ex['decay']),
        }

    def _synthesise(self, params_np: dict) -> np.ndarray:
        config = SynthConfig(
            num_samples=self.num_audio_samples,
            fs=self.fs,
            lagrange_order=self.lagrange_order,
            training=False,
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