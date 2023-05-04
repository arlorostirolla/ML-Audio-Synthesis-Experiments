import librosa, math
import torch, torchaudio.transforms as T
from torchsynth.module import (MonophonicKeyboard, Oscillator,  
                                ADSREnvelope, SVFilter,VCA,LFO,PulseWidthModulationOscillator,
                                Noise,LowpassFilter,HighpassFilter,Phaser,Delay,Chorus)

from torchaudio.functional import compute_deltas
from torch.optim import SGD
import torchaudio.functional as F
from Classes import *




class FlexibleSynthesizer(torch.nn.Module):
    def __init__(self, sample_rate: int = 44100, num_oscillators: int = 6, num_filters: int = 6, num_blend_methods: int = 3):
        super().__init__()
        self.sample_rate = sample_rate
        self.keyboard = MonophonicKeyboard()
        self.oscillators = torch.nn.ModuleList([
            Oscillator('sine'),
            Oscillator('square'),
            Oscillator('sawtooth'),
            Oscillator('triangle'),
            PulseWidthModulationOscillator(),
            Noise(),
            WhiteNoiseOscillator(),
            PinkNoiseOscillator()
        ][:num_oscillators])
        self.envelope = ADSREnvelope()
        self.filters = torch.nn.ModuleList([
            SVFilter(),
            LowpassFilter(),
            HighpassFilter(),
        ][:num_filters])
        self.vca = VCA()
        self.lfo = LFO()
        self.phaser = Phaser()
        self.delay = Delay()
        self.chorus = Chorus()
        self.reverb = T.Reverb(sample_rate)
        self.bitcrusher = T.MuLawEncoding()
        self.tremolo = Tremolo(sample_rate)
        self.distortion = TanhDistortion()
        self.flanger = Flanger(sample_rate)
        self.stereo_widener = StereoWidener(sample_rate)
        self.notch_filter = NotchFilter(sample_rate)
        self.comb_filter = CombFilter(sample_rate)
        self.bandpass_filter = BandpassFilter(sample_rate)

                # Define learnable parameters
        self.osc_type = torch.nn.Parameter(torch.randint(num_oscillators, size=(1,)))
        self.osc_type2 = torch.nn.Parameter(torch.randint(num_oscillators, size=(1,)))
        self.filter_type = torch.nn.Parameter(torch.randint(num_filters, size=(1,)))
        self.blend_method = torch.nn.Parameter(torch.randint(num_blend_methods, size=(1,)))


    def forward(self, midi_f0, duration, note_on_duration, adsr_params, filter_params, osc_type, filter_type, osc_type2=None, blend_method=None):
        audio = self.keyboard(midi_f0, duration, self.sample_rate)
        
        osc_audio = 0.0
        for i, oscillator in enumerate(self.oscillators):
            if i == self.osc_type or i == self.osc_type2:
                osc_audio += oscillator(audio, self.sample_rate)
        
        envelope_audio = self.envelope(adsr_params, note_on_duration, osc_audio, self.sample_rate)

        filtered_audio = envelope_audio
        for i, f in enumerate(self.filters):
            if i == self.filter_type:
                filtered_audio = f(filtered_audio, filter_params)

        processed_audio = filtered_audio
        if self.blend_method == 0:
            processed_audio += osc_audio
        elif self.blend_method == 1:
            processed_audio = osc_audio * filtered_audio
        elif self.blend_method == 2:
            processed_audio = osc_audio * (1 + self.lfo(filtered_audio, self.sample_rate))
        
        audio = self.envelope(audio, note_on_duration, adsr_params)
        audio = self.filters[filter_type](audio, filter_params)
        audio = self.vca(audio)
        audio = self.lfo(audio)
        audio = self.phaser(audio)
        audio = self.delay(audio)
        audio = self.chorus(audio)
        audio = self.reverb(audio.unsqueeze(0)).squeeze(0)
        audio = self.tremolo(audio)
        audio = self.bitcrusher(audio.unsqueeze(0)).squeeze(0)
        audio = self.distortion(audio)
        audio = self.flanger(audio, self.sample_rate)
        audio = self.stereo_widener(audio, self.sample_rate)
        audio = self.notch_filter(audio, self.sample_rate)
        audio = self.comb_filter(audio)
        audio = self.bandpass_filter(audio, self.sample_rate)

        return audio
    

loss_function = torch.nn.MSELoss()


def optimize_synthesizer(synth, target_audio, num_iterations, learning_rate):
    # Convert target audio to a torch tensor
    target_audio = torch.tensor(target_audio, dtype=torch.float32)
    
    # Create the optimizer
    optimizer = SGD(synth.parameters(), lr=learning_rate)
    
    # Optimization loop
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Generate the synthesizer output using some input parameters
        midi_f0 = torch.tensor([60.0], dtype=torch.float32)
        duration = torch.tensor([1.0], dtype=torch.float32)
        note_on_duration = torch.tensor([1.0], dtype=torch.float32)
        adsr_params = torch.tensor([0.1, 0.1, 0.5, 0.1], dtype=torch.float32)
        filter_params = torch.tensor([1000.0, 1.0], dtype=torch.float32)
        osc_type = 0  # This can be changed to select a different oscillator
        filter_type = 0  # This can be changed to select a different filter
        osc_type2 = 1  # This can be changed to select a second oscillator
        blend_method = 'sum'  # This can be changed to select a different blending method (sum, multiply, or max)
        synth_output = synth(midi_f0, duration, note_on_duration, adsr_params, filter_params, osc_type, filter_type, osc_type2, blend_method)
        
        # Calculate the loss
        loss = loss_function(synth_output, target_audio)
        
        # Backpropagate and update the synthesizer's parameters
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {i + 1}: Loss = {loss.item()}")


if __name__ == '__main__':
    target_audio_file = "528491.wav"
    target_audio, sr = librosa.load(target_audio_file, sr=None, mono=True)  
    synth = FlexibleSynthesizer()
    optimize_synthesizer(synth, target_audio, num_iterations=1000, learning_rate=0.01)