import torch.nn.functional as F
import torch
import torch.optim as optim
from Classes import *
import torch.nn as nn
from torch.distributions import Gumbel
import torchaudio

class ModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class DifferentiableSynth(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = nn.ModuleList([ModuleWrapper(module) for module in modules])
        num_modules = len(modules)
        self.conn_weights = nn.Parameter(torch.randn(num_modules, num_modules + 1))

    def forward(self, midi_f0, duration, sample_rate):
        output_audio = torch.zeros(int(duration * sample_rate))
        for i, module in enumerate(self.modules):
            output_audio += module(midi_f0, duration, sample_rate) * self.conn_weights[i, -1]
            for j, next_module in enumerate(self.modules):
                if i != j:
                    output_audio = next_module(output_audio) * self.conn_weights[i, j]
        return output_audio


    def set_temperature(self, temperature):
        self.temperature = max(temperature, 1e-3)


if __name__ == '__main__':
    # Load target audio
    target_filename = './528491.wav'
    target_waveform, _ = torchaudio.load(target_filename)
    target_audio = target_waveform[0]

    # Create a differentiable synthesizer
    modules = [WhiteNoiseOscillator(), PinkNoiseOscillator(), BandpassFilter(), Flanger()]
    diff_synth = DifferentiableSynth(modules)

    # Set optimizer
    optimizer = optim.Adam(diff_synth.parameters(), lr=0.001)

    # Optimize the synthesizer
    midi_f0 = 60
    duration = 1.0
    iterations = 1000
    loss_fn = F.mse_loss

    for _ in range(iterations):
        optimizer.zero_grad()
        output_audio = diff_synth(midi_f0, duration, sample_rate=44100)
        loss = loss_fn(output_audio, target_audio)
        loss.backward()
        optimizer.step()