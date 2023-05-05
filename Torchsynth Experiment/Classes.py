import torch, torchaudio.transforms as T
from torchaudio.functional import compute_deltas
import math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, oscillators):
        super(FMSynthesis, self).__init__()
        self.oscillators = nn.ModuleList(oscillators)
        num_oscillators = len(oscillators)
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
    def __init__(self, fundamental_freq=440.0):
        super(HammondOrgan, self).__init__()
        self.fundamental_freq = nn.Parameter(torch.tensor(fundamental_freq, dtype=torch.float32))
        self.harmonics_amplitudes = nn.Parameter(torch.ones(9, dtype=torch.float32))

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate

        output_audio = torch.zeros(int(duration * sample_rate))
        for i, amplitude in enumerate(self.harmonics_amplitudes):
            harmonic_freq = (i + 1) * self.fundamental_freq
            harmonic_phase = 2 * torch.pi * harmonic_freq * timesteps
            harmonic_wave = amplitude * torch.sin(harmonic_phase)
            output_audio += harmonic_wave

        return output_audio

class WhiteNoiseOscillator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, midi_f0, duration, sample_rate):
        return torch.randn(int(duration * sample_rate))

class PinkNoiseOscillator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, midi_f0, duration, sample_rate):
        white_noise = torch.randn(duration * sample_rate)
        return compute_deltas(white_noise, win_length=2)

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
        time = torch.arange(0, audio.shape[-1], dtype=torch.float32) / self.sample_rate
        lfo = 0.5 * self.depth * (1 + torch.sin(2 * math.pi * self.rate * time))
        delay_samples = (self.delay_time + lfo).clamp(0) * self.sample_rate
        return torch.roll(audio, int(delay_samples), dims=-1)

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
        max_delay = len(self.impulse_response) - 1
        audio = torch.cat([audio, torch.zeros(max_delay)])
        audio = torch.conv1d(audio.unsqueeze(0), self.impulse_response.unsqueeze(0).unsqueeze(0)).squeeze(0)
        audio = audio / torch.max(torch.abs(audio))

        return audio

class Delay(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delay_time = torch.nn.Parameter(torch.tensor(0.1))
        self.feedback = torch.nn.Parameter(torch.tensor(0.5))
        self.mix = torch.nn.Parameter(torch.tensor(0.5))
        self.sample_rate = 44100

    def forward(self, audio):
        delay_samples = int(self.delay_time * self.sample_rate)
        audio_delayed = torch.cat([torch.zeros(delay_samples), audio])
        audio = audio + self.feedback * audio_delayed[:len(audio)]
        audio = self.mix * audio + (1 - self.mix) * audio_delayed[:len(audio)]
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
        lfo = 0.5 * (1 + torch.sin(2 * np.pi * (1 / self.delay_time) * torch.arange(len(audio)) / self.sample_rate))
        offset = int(self.depth * len(audio))
        audio_delayed = torch.cat([audio[:offset], audio[:-offset]])
        audio = audio + self.feedback * audio_delayed * lfo
        audio = self.mix * audio + (1 - self.mix) * audio_delayed
        return audio
    
class BandpassFilter(nn.Module):
    def __init__(self, center_freq, bandwidth, sample_rate):
        super(BandpassFilter, self).__init__()
        self.sample_rate = sample_rate
        self.center_freq = nn.Parameter(torch.tensor(center_freq, dtype=torch.float32))
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float32))
        self.gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        low_freq = self.center_freq - self.bandwidth / 2
        high_freq = self.center_freq + self.bandwidth / 2
        x_low = F.lfilter(x, *torch.butter(1, low_freq / (self.sample_rate / 2), btype='high'))
        x_band = F.lfilter(x_low, *torch.butter(1, high_freq / (self.sample_rate / 2), btype='low'))
        return x_band * self.gain

class Equalizer(nn.Module):
    def __init__(self, num_bands, sample_rate):
        super(Equalizer, self).__init__()
        self.sample_rate = sample_rate
        self.filters = nn.ModuleList()
        for _ in range(num_bands):
            self.filters.append(BandpassFilter(1000.0, 200.0, sample_rate))

    def forward(self, x):
        output = torch.zeros_like(x)
        for band_filter in self.filters:
            output += band_filter(x)
        return output
    
class LFO(nn.Module):
    def __init__(self, frequency=1.0, amplitude=1.0, waveform=None):
        super(LFO, self).__init__()
        self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))
        
        if waveform is None:
            self.waveform = torch.sin
        else:
            self.waveform = waveform

    def forward(self, duration, sample_rate):
        timesteps = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
        phase = 2 * torch.pi * self.frequency * timesteps
        lfo_wave = self.amplitude * self.waveform(phase)
        return lfo_wave
    
class Compressor(nn.Module):
    def __init__(self, threshold=-20.0, ratio=4.0, attack=0.01, release=0.1):
        super(Compressor, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.ratio = nn.Parameter(torch.tensor(ratio, dtype=torch.float32))
        self.attack = nn.Parameter(torch.tensor(attack, dtype=torch.float32))
        self.release = nn.Parameter(torch.tensor(release, dtype=torch.float32))

    def forward(self, audio, sample_rate):
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-5)
        gain_reduction_db = torch.clamp((self.threshold - audio_db) * (1.0 - 1.0 / self.ratio), min=0)
        
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
    def __init__(self, depth=5, rate=5):
        super(Vibrato, self).__init__()
        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.float32))
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float32))

    def forward(self, audio, sample_rate):
        max_delay_samples = int(self.depth * sample_rate / 1000)
        delayed_audio = torch.zeros_like(audio)
        delayed_audio[max_delay_samples:] = audio[:-max_delay_samples]

        timesteps = torch.arange(len(audio), dtype=torch.float32) / sample_rate
        lfo = 0.5 * (1 + torch.sin(2 * torch.pi * self.rate * timesteps))

        delay_range = torch.arange(max_delay_samples, dtype=torch.float32)
        delay_samples = torch.round((self.depth * lfo).unsqueeze(-1) + delay_range).long()
        delayed_audio_modulated = torch.gather(delayed_audio, 0, delay_samples).mean(dim=1)

        return delayed_audio_modulated

def one_hot(index, size):
    one_hot_vector = np.zeros(size)
    one_hot_vector[index] = 1
    return one_hot_vector

oscillators = [SineWaveOscillator, TriangleWaveOscillator, SquareWaveOscillator, SawtoothWaveOscillator, 
               WhiteNoiseOscillator, PinkNoiseOscillator, PulseWaveOscillator, FMSynthesis, HammondOrgan]

effects = [Tremolo, Phaser, Flanger, Chorus, Delay, Reverb, StereoWidener, Equalizer, TanhDistortion, LFO, 
           Compressor, Vibrato]


num_oscillators = len(oscillators)
num_effects = len(effects)
num_configurations = num_oscillators + num_effects

oscillator_one_hot = {oscillator: one_hot(i, num_configurations) for i, oscillator in enumerate(oscillators)}
effect_one_hot = {effect: one_hot(i + num_oscillators, num_configurations) for i, effect in enumerate(effects)}

print("Oscillator one-hot encodings:")
for k, v in oscillator_one_hot.items():
    print(f"{k}: {v}")

print("\nEffect one-hot encodings:")
for k, v in effect_one_hot.items():
    print(f"{k}: {v}")