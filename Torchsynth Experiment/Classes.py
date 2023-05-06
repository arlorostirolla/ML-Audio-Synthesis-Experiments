import torch, torchaudio.transforms as T
from torchaudio.functional import compute_deltas
import math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from scipy.signal import butter

class SineWaveOscillator(nn.Module):
    def __init__(self, frequency=440.0, amplitude=1.0):
        super(SineWaveOscillator, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
        sine_wave = self.amplitude * torch.sin(2 * torch.pi * self.frequency * timesteps)
        return sine_wave
    
class TriangleWaveOscillator(nn.Module):
    def __init__(self, frequency=440.0, amplitude=1.0):
        super(TriangleWaveOscillator, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
        phase = 2 * torch.pi * self.frequency * timesteps
        triangle_wave = 2 / torch.pi * torch.asin(torch.sin(phase))
        return self.amplitude * triangle_wave

class SquareWaveOscillator(nn.Module):
    def __init__(self, frequency=440.0, amplitude=1.0):
        super(SquareWaveOscillator, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
        phase = 2 * torch.pi * self.frequency * timesteps
        square_wave = self.amplitude * torch.sign(torch.sin(phase))
        return square_wave
    
class SawtoothWaveOscillator(nn.Module):
    def __init__(self, frequency=440.0, amplitude=1.0):
        super(SawtoothWaveOscillator, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
        phase = 2 * torch.pi * self.frequency * timesteps
        sawtooth_wave = self.amplitude * (2 / torch.pi * (phase - torch.pi))
        return sawtooth_wave

class PulseWaveOscillator(nn.Module):
    def __init__(self, frequency=440.0, amplitude=1.0, pulse_width=0.5):
        super(PulseWaveOscillator, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))
        self.pulse_width = nn.Parameter(torch.tensor(pulse_width, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
        phase = 2 * torch.pi * self.frequency * timesteps
        pulse_wave = self.amplitude * torch.where(torch.sin(phase) < self.pulse_width, torch.tensor(1.0), torch.tensor(-1.0))
        return pulse_wave
    
class FMSynthesis(nn.Module):
    def __init__(self):
        super(FMSynthesis, self).__init__()
        self.oscillators = nn.ModuleList([SineWaveOscillator(), TriangleWaveOscillator(),
                                          SquareWaveOscillator(),SawtoothWaveOscillator(),
                                          PulseWaveOscillator()])
        num_oscillators = len(self.oscillators)
        self.combination_weights = nn.Parameter(torch.randn(num_oscillators, num_oscillators))

    def forward(self, duration, sample_rate):
        output_audio = torch.zeros(int(duration * sample_rate))
        for i, carrier in enumerate(self.oscillators):
            modulator_audio = torch.zeros(int(duration * sample_rate))
            for j, modulator in enumerate(self.oscillators):
                if i != j:
                    modulator_audio += modulator(duration, sample_rate) * self.combination_weights[i, j]

            modulator_phase = 2 * torch.pi * modulator_audio
            modulated_carrier = torch.sin(2 * torch.pi * carrier.frequency * duration + modulator_phase)
            output_audio += carrier.amplitude * modulated_carrier

        return output_audio
    
class HammondOrgan(nn.Module):
    def __init__(self, frequency=440.0):
        super(HammondOrgan, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.harmonics_amplitudes = nn.Parameter(torch.ones(9, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate

        output_audio = torch.zeros(int(duration * sample_rate))
        for i, amplitude in enumerate(self.harmonics_amplitudes):
            harmonic_freq = (i + 1) * self.frequency
            harmonic_phase = 2 * torch.pi * harmonic_freq * timesteps
            harmonic_wave = amplitude * torch.sin(harmonic_phase)
            output_audio += harmonic_wave

        return output_audio

class WhiteNoiseOscillator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, duration, sample_rate):
        return torch.randn(int(duration * sample_rate))

class PinkNoiseOscillator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, duration, sample_rate):
        white_noise = torch.randn(duration * sample_rate)
        return compute_deltas(white_noise, win_length=3)

class TanhDistortion(torch.nn.Module):
    def __init__(self, init_gain: float = 2.0):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(init_gain))

    def forward(self, audio):
        return torch.tanh(self.gain * audio)

class Flanger(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, init_delay_time: float = 0.002, init_depth: float = 0.005, init_rate: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.delay_time = torch.nn.Parameter(torch.tensor(init_delay_time))
        self.depth = torch.nn.Parameter(torch.tensor(init_depth))
        self.rate = torch.nn.Parameter(torch.tensor(init_rate))

    def forward(self, audio):
        if audio.size(0) == 0:
            return audio

        time = torch.arange(0, audio.shape[-1], dtype=torch.float32) / self.sample_rate
        lfo = 0.5 * self.depth * (1 + torch.sin(2 * math.pi * self.rate * time))
        delay_samples = (self.delay_time + lfo).clamp(0) * self.sample_rate

        # Round delay_samples to the nearest integer
        delay_samples_rounded = torch.round(delay_samples).long()

        # Roll each time step along the time axis
        rolled_audio = torch.stack([torch.roll(channel, shift, dims=-1) for channel, shift in zip(audio, delay_samples_rounded)])

        return rolled_audio



class Tremolo(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, init_depth: float = 0.5, init_rate: float = 5.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.depth = torch.nn.Parameter(torch.tensor(init_depth))
        self.rate = torch.nn.Parameter(torch.tensor(init_rate))
    
    def forward(self, audio):
        time = torch.arange(0, audio.shape[-1], dtype=torch.float32) / self.sample_rate
        lfo = 0.5 * self.depth * (1 + torch.sin(2 * math.pi * self.rate * time))
        return audio * (1 - lfo)

class StereoWidener(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, init_delay_time: float = 0.005):
        super().__init__()
        self.sample_rate = sample_rate
        self.delay_time = torch.nn.Parameter(torch.tensor(init_delay_time))

    def forward(self, audio):
        delay_samples = int(self.sample_rate * self.delay_time)
        delayed_audio = torch.roll(audio, delay_samples, dims=-1)
        stereo_audio = torch.stack((audio, delayed_audio), dim=0)
        return stereo_audio
    
class Filter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cutoff = torch.nn.Parameter(torch.tensor(1000.0))
        self.resonance = torch.nn.Parameter(torch.tensor(1.0))


    def forward(self, audio, filter_params, mode):
        self.cutoff = filter_params['cutoff']
        self.resonance = filter_params['resonance']
        self.mode = filter_params['mode']

        if mode == 'lowpass':
            a, b = self.lowpass_coefficients()
        elif mode == 'highpass':
            a, b = self.highpass_coefficients()
        elif mode == 'bandpass':
            a, b = self.bandpass_coefficients()
        else:
            raise ValueError('Invalid mode: %s' % self.mode)

        y = torch.zeros_like(audio)
        for n in range(len(audio)):
            y[n] = b[0] * audio[n] + b[1] * self.z1 - a[1] * y[n - 1]
            self.z1 = b[2] * audio[n] - a[2] * y[n - 1]
        return y

    def lowpass_coefficients(self):
        omega = 2 * np.pi * self.cutoff / self.sample_rate
        alpha = np.sin(omega) / (2 * self.resonance)
        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha
        b0 = (1 - np.cos(omega)) / 2
        b1 = 1 - np.cos(omega)
        b2 = (1 - np.cos(omega)) / 2
        return np.array([a0, a1, a2]) / a0, np.array([b0, b1, b2]) / a0

    def highpass_coefficients(self):
        omega = 2 * np.pi * self.cutoff / self.sample_rate
        alpha = np.sin(omega) / (2 * self.resonance)
        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha
        b0 = (1 + np.cos(omega)) / 2
        b1 = -(1 + np.cos(omega))
        b2 = (1 + np.cos(omega)) / 2
        return np.array([a0, a1, a2]) / a0, np.array([b0, b1, b2]) / a0

    def bandpass_coefficients(self):
        omega = 2 * np.pi * self.cutoff / self.sample_rate
        alpha = np.sin(omega) / (2 * self.resonance)
        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha
        b0 = np.sin(omega) / 2
        b1 = 0
        b2 = -np.sin(omega) / 2
        return np.array([a0, a1, a2]) / a0, np.array([b0, b1, b2]) / a0
    
class Phaser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.5))
        self.depth = torch.nn.Parameter(torch.tensor(0.5))
        self.feedback = torch.nn.Parameter(torch.tensor(0.5))
        self.mix = torch.nn.Parameter(torch.tensor(0.5))
        self.sample_rate = 44100

    def forward(self, audio):
        lfo = 0.5 * (1 + torch.sin(2 * np.pi * self.rate * torch.arange(len(audio)) / self.sample_rate))
        offset = int(self.depth * len(audio))
        audio_delayed = torch.cat([audio[:offset], audio[:-offset]])
        audio = audio + self.feedback * audio_delayed * lfo
        audio = self.mix * audio + (1 - self.mix) * audio_delayed
        return audio
    
class Reverb(torch.nn.Module):
    def __init__(self, sample_rate, max_duration=5.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.impulse_response = torch.nn.Parameter(torch.randn((int(self.sample_rate * self.max_duration),)))

    def forward(self, audio):
        if audio.size(0) == 0:
            return audio

        max_delay = len(self.impulse_response) - 1
        zeros = torch.zeros(audio.size(0), max_delay)
        audio = torch.cat([audio, zeros], dim=-1)

        if audio.size(0) == 1:  # Handling single channel audio
            audio = torch.conv1d(audio.unsqueeze(0), self.impulse_response.unsqueeze(0).unsqueeze(0)).squeeze(0)
        else:  # Handling multi-channel audio
            audio = torch.conv1d(audio, self.impulse_response.unsqueeze(0).unsqueeze(0))

        audio = audio / torch.max(torch.abs(audio))
        return audio




class Delay(torch.nn.Module):
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.delay_time = torch.nn.Parameter(torch.tensor(0.1))
        self.feedback = torch.nn.Parameter(torch.tensor(0.5))
        self.mix = torch.nn.Parameter(torch.tensor(0.5))
        self.sample_rate = sample_rate

    def forward(self, audio):
        delay_samples = int(self.delay_time * self.sample_rate)
        zeros = torch.zeros(audio.size(0), delay_samples)
        audio_delayed = torch.cat([zeros, audio], dim=-1)
        audio = audio + self.feedback * audio_delayed[:, :audio.size(1)]
        audio = self.mix * audio + (1 - self.mix) * audio_delayed[:, :audio.size(1)]
        return audio
    
class Chorus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delay_time = torch.nn.Parameter(torch.tensor(0.05))
        self.depth = torch.nn.Parameter(torch.tensor(0.01))
        self.feedback = torch.nn.Parameter(torch.tensor(0.5))
        self.mix = torch.nn.Parameter(torch.tensor(0.5))
        self.sample_rate = 44100

    def forward(self, audio):
        if audio.nelement() == 0:
            return audio
        lfo = 0.5 * (1 + torch.sin(2 * np.pi * (1 / self.delay_time) * torch.arange(len(audio)) / self.sample_rate))

        # Calculate offset as a fraction of audio length and use modulo to keep it within bounds
        offset = int(self.depth * len(audio)) % len(audio)

        audio_delayed = torch.cat([audio[:offset], audio[:-offset]])

        # Expand lfo tensor to match the shape of audio and audio_delayed
        lfo = lfo.view(1, -1).expand_as(audio)

        audio = audio + self.feedback * audio_delayed * lfo
        audio = self.mix * audio + (1 - self.mix) * audio_delayed
        return audio


    
class BandpassFilter(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.low_cutoff = nn.Parameter(torch.tensor(200.0))
        self.high_cutoff = nn.Parameter(torch.tensor(1000.0))

    def forward(self, x):
        low_freq = torch.clamp(self.low_cutoff, 0, self.sample_rate / 2)
        high_freq = torch.clamp(self.high_cutoff, low_freq.item(), self.sample_rate / 2)

        b_high, a_high = butter(1, low_freq.detach().item() / (self.sample_rate / 2), btype='high')
        b_low, a_low = butter(1, high_freq.detach().item() / (self.sample_rate / 2), btype='low')

        b_high, a_high = torch.tensor(b_high).float(), torch.tensor(a_high).float()
        b_low, a_low = torch.tensor(b_low).float(), torch.tensor(a_low).float()

        x_high = AF.lfilter(x, a_high, b_high)
        x_low = AF.lfilter(x, a_low, b_low)

        return x_high + x_low

class Equalizer(nn.Module):
    def __init__(self, num_bands, sample_rate):
        super(Equalizer, self).__init__()
        self.sample_rate = sample_rate
        self.filters = nn.ModuleList()
        for _ in range(num_bands):
            self.filters.append(BandpassFilter(sample_rate))

    def forward(self, x):
        output = torch.zeros_like(x)
        for band_filter in self.filters:
            output += band_filter(x)
        return output
    
    
class Compressor(nn.Module):
    def __init__(self, threshold=-20.0, ratio=4.0, attack=0.01, release=0.1, sample_rate=44100):
        super(Compressor, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.ratio = nn.Parameter(torch.tensor(ratio, dtype=torch.float32))
        self.attack = nn.Parameter(torch.tensor(attack, dtype=torch.float32))
        self.release = nn.Parameter(torch.tensor(release, dtype=torch.float32))

    def forward(self, audio):
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-5)
        gain_reduction_db = torch.clamp((self.threshold - audio_db) * (1.0 - 1.0 / self.ratio), min=0)
        if gain_reduction_db.numel() == 0:
            return audio
        smoothed_gain_reduction_db = torch.zeros_like(gain_reduction_db)
        gain_reduction = 0
        for i in range(len(gain_reduction_db)):
            if gain_reduction_db[i] > gain_reduction:
                coeff = self.attack
            else:
                coeff = self.release
            gain_reduction += (1 - coeff) * (gain_reduction_db[i] - gain_reduction)
            smoothed_gain_reduction_db[i] = gain_reduction

        gain_reduction_linear = 10 ** (-smoothed_gain_reduction_db / 20)
        compressed_audio = audio * gain_reduction_linear

        return compressed_audio

class Vibrato(nn.Module):
    def __init__(self, depth=5, rate=5, sample_rate=44100):
        super(Vibrato, self).__init__()
        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.float32))
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float32))
        self.sample_rate = sample_rate

    def forward(self, audio):
        batch_size, num_channels, num_samples = audio.shape

        max_delay_samples = int(self.depth * self.sample_rate / 1000)
        delayed_audio = torch.zeros_like(audio)
        delayed_audio[:, :, max_delay_samples:] = audio[:, :, :-max_delay_samples]

        timesteps = torch.arange(num_samples, dtype=torch.float32, device=audio.device) / self.sample_rate
        lfo = 0.5 * (1 + torch.sin(2 * torch.pi * self.rate * timesteps))

        delay_range = torch.arange(max_delay_samples, dtype=torch.float32, device=audio.device)
        delay_samples = (self.depth * lfo).unsqueeze(1) + delay_range.unsqueeze(0)
        delay_samples = delay_samples.unsqueeze(0).expand(batch_size, num_channels, -1)
        delayed_audio_modulated = torch.gather(delayed_audio, 2, delay_samples.long().unsqueeze(-1)).mean(dim=2)

        return delayed_audio_modulated

class Volume(nn.Module):
    def __init__(self, sample_rate):
        super(Volume, self).__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.gain + self.bias
    
class ADSR(nn.Module):
    def __init__(self):
        super(ADSR, self).__init__()
        self.attack = nn.Parameter(torch.tensor(0.1))
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.sustain = nn.Parameter(torch.tensor(0.5))
        self.release = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        length = x.size(1)
        envelope = torch.zeros_like(x)

        attack_samples = int(length * self.attack.item())
        decay_samples = int(length * self.decay.item())
        release_samples = int(length * self.release.item())

        envelope[:, :attack_samples] = torch.linspace(0, 1, attack_samples).unsqueeze(0)
        envelope[:, attack_samples:attack_samples+decay_samples] = torch.linspace(1, self.sustain.item(), decay_samples).unsqueeze(0)
        envelope[:, -release_samples:] = torch.linspace(self.sustain.item(), 0, release_samples).unsqueeze(0)

        return x * envelope


oscillators = [SineWaveOscillator, TriangleWaveOscillator, SquareWaveOscillator, SawtoothWaveOscillator, 
               WhiteNoiseOscillator, PinkNoiseOscillator, PulseWaveOscillator, FMSynthesis, HammondOrgan]

effects = [Volume, ADSR, Tremolo, Phaser, Flanger, Chorus, Delay, Reverb, StereoWidener, Equalizer, TanhDistortion, 
           Compressor, Vibrato]


from torch.nn.parameter import Parameter

class CustomSynthesizer(nn.Module):
    def __init__(self, num_fm_synths=1, duration=5, sample_rate=44100):
        super(CustomSynthesizer, self).__init__()
        self.duration = duration
        self.sample_rate = sample_rate

        # Instantiate oscillators
        self.oscillators = nn.ModuleList([
            SineWaveOscillator(),
            TriangleWaveOscillator(),
            SquareWaveOscillator(),
            SawtoothWaveOscillator(),
            WhiteNoiseOscillator(),
            PinkNoiseOscillator(),
            PulseWaveOscillator(),
            *([FMSynthesis() for _ in range(num_fm_synths)]),
            HammondOrgan()
        ])

        # Instantiate effects
        self.effects = nn.ModuleDict({
            'volume': Volume(self.sample_rate),
             #'adsr': ADSR(),
            'tremolo': Tremolo(),
            'phaser': Phaser(),
            'flanger': Flanger(),
            'chorus': Chorus(),
            'delay': Delay(),
            'reverb': Reverb(self.sample_rate),
            'stereo_widener': StereoWidener(),
            'equalizer': Equalizer(10, self.sample_rate),
            'tanh_distortion': TanhDistortion(),
            'compressor': Compressor(self.sample_rate)
        })

        # Learnable volume parameters for oscillators
        self.volume_params = Parameter(torch.ones(len(self.oscillators)))

        # Learnable effect chain parameters for each oscillator
        num_effects = len(self.effects)
        self.effect_chain_params = Parameter(torch.rand(len(self.oscillators), num_effects))

    def apply_effects(self, osc_idx, effect_probs):
        audio = self.oscillators[osc_idx].forward(self.duration, self.sample_rate)
        # Add a batch dimension to the input waveform
        #audio = audio.unsqueeze(0)

        for effect, prob in zip(self.effects.values(), effect_probs):
            print(effect)
            processed_audio = effect.forward(audio)
            audio = audio * (1 - prob) + processed_audio * prob

        return audio

    def mix_oscillators(self):
        mixed_audio = torch.zeros(self.duration * self.sample_rate)
        effect_probabilities = torch.softmax(self.effect_chain_params, dim=-1)

        for osc_idx, (osc, vol_param, effect_probs) in enumerate(zip(self.oscillators, self.volume_params, effect_probabilities)):
            audio_with_effects = self.apply_effects(osc_idx, effect_probs)
            mixed_audio += vol_param * audio_with_effects

        return mixed_audio

    def forward(self):
        return self.mix_oscillators()