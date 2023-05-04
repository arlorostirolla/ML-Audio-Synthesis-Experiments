import torch, torchaudio.transforms as T
from torchaudio.functional import compute_deltas
import math

def blend_signals(audio1, audio2, method):
    if method == 'sum':
        return audio1 + audio2
    elif method == 'multiply':
        return audio1 * audio2
    elif method == 'max':
        return torch.max(audio1, audio2)
    else:
        raise ValueError("Invalid blending method")
    
class WhiteNoiseOscillator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, midi_f0, duration, sample_rate):
        return torch.randn(duration * sample_rate)

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
    
class BandpassFilter(torch.nn.Module):
    def __init__(self,sample_rate: int = 44100, init_cutoff_low: float = 200.0, init_cutoff_high: float = 5000.0):
        super().__init__()
        self.cutoff_low = torch.nn.Parameter(torch.tensor(init_cutoff_low))
        self.cutoff_high = torch.nn.Parameter(torch.tensor(init_cutoff_high))

    def forward(self, audio, sample_rate):
        audio_low = F.lowpass_biquad(audio, sample_rate, self.cutoff_low)
        audio_high = F.highpass_biquad(audio, sample_rate, self.cutoff_high)
        return audio_low - audio_high

class NotchFilter(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, init_cutoff: float = 1000.0, init_q: float = 0.707):
        super().__init__()
        self.cutoff = torch.nn.Parameter(torch.tensor(init_cutoff))
        self.q = torch.nn.Parameter(torch.tensor(init_q))

    def forward(self, audio, sample_rate):
        return F.bandstop_biquad(audio, sample_rate, self.cutoff, self.q)

class CombFilter(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, init_delay_time: float = 0.01, init_feedback: float = 0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.delay_time = torch.nn.Parameter(torch.tensor(init_delay_time))
        self.feedback = torch.nn.Parameter(torch.tensor(init_feedback))

    def forward(self, audio):
        delay_samples = int(self.sample_rate * self.delay_time)
        delayed_audio = torch.roll(audio, delay_samples, dims=-1)
        return audio + self.feedback * delayed_audio

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