import numpy as np
from abc import ABC, abstractmethod
from effects_and_functions import *

SAMPLE_RATE = 44100
FREQUENCY = 440
DURATION = 5  # in seconds
AMPLITUDE = 2 ** 15 - 1
AUDIO_TYPE = "S16LE"  # 16-bit signed little-endian PCM


class Synthesizer(ABC):
    def __init__(self, sample_rate=SAMPLE_RATE, frequency=FREQUENCY, duration=DURATION, amplitude=AMPLITUDE):
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.duration = duration
        self.amplitude = amplitude

    @abstractmethod
    def synthesize(self):
        pass

class FMSynthesizer(Synthesizer):

    def __init__(self, modulator_frequency_ratio, modulator_amplitude, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulator_frequency_ratio = modulator_frequency_ratio
        self.modulator_amplitude = modulator_amplitude

    def synthesize(self):
        t = np.linspace(0, self.duration, self.duration * self.sample_rate, False)
        modulator = np.sin(2 * np.pi * self.frequency * self.modulator_frequency_ratio * t)
        carrier = np.sin(2 * np.pi * self.frequency * t + self.modulator_amplitude * modulator)
        return (self.amplitude * carrier).astype(np.int16)


class GranularSynthesizer(Synthesizer):

    def __init__(self, grain_duration, overlap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grain_duration = grain_duration
        self.overlap = overlap

    def synthesize(self):
        t = np.linspace(0, self.grain_duration, int(self.grain_duration * self.sample_rate), False)
        grain = self.amplitude * np.sin(2 * np.pi * self.frequency * t)

        num_samples = int(self.duration * self.sample_rate)
        step_size = int(self.grain_duration * self.sample_rate * (1 - self.overlap))
        result = np.zeros(num_samples, dtype=np.int16)

        for i in range(0, num_samples, step_size):
            result[i:i + len(grain)] += grain

        return result


class AdditiveSynthesizer(Synthesizer):

    def __init__(self, harmonics_amplitudes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harmonics_amplitudes = harmonics_amplitudes

    def synthesize(self):
        t = np.linspace(0, self.duration, self.duration * self.sample_rate, False)
        result = np.zeros_like(t)

        for i, amp in enumerate(self.harmonics_amplitudes):
            harmonic = (i + 1) * self.frequency
            result += amp * np.sin(2 * np.pi * harmonic * t)

        return (self.amplitude * result).astype(np.int16)

class WavetableSynthesizer(Synthesizer):

    def __init__(self, wavetables, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wavetables = wavetables

    def synthesize(self):
        num_wavetables = len(self.wavetables)
        wavetable_size = len(self.wavetables[0])
        t = np.linspace(0, self.duration, self.duration * self.sample_rate, False)

        # Interpolate between wavetables based on frequency
        wavetable_idx = int(np.interp(self.frequency, [0, self.sample_rate / 2], [0, num_wavetables - 1]))

        if wavetable_idx < num_wavetables - 1:
            alpha = (self.frequency - self.wavetables[wavetable_idx].frequency) / (self.wavetables[wavetable_idx + 1].frequency - self.wavetables[wavetable_idx].frequency)
            wavetable = (1 - alpha) * self.wavetables[wavetable_idx] + alpha * self.wavetables[wavetable_idx + 1]
        else:
            wavetable = self.wavetables[wavetable_idx]

        # Generate waveform using the selected wavetable
        phase = (t * self.frequency) % 1
        waveform = np.interp(phase * wavetable_size, np.arange(wavetable_size), wavetable)

        return (self.amplitude * waveform).astype(np.int16)

class ModularSynthesizer:
    def __init__(self, sample_rate=44100):
        self.modules = {}
        self.connections = []
        self.sample_rate = sample_rate

    def add_module(self, name, module):
        self.modules[name] = module

    def connect(self, source_name, target_name, source_output=0, target_input=0):
        self.connections.append((source_name, target_name, source_output, target_input))

    def synthesize(self, duration):
        # Process each module in the synthesizer
        for name, module in self.modules.items():
            module.process(duration, self.sample_rate)

        # Apply the connections between modules
        for source_name, target_name, source_output, target_input in self.connections:
            source_module = self.modules[source_name]
            target_module = self.modules[target_name]
            source_signal = source_module.get_output(source_output)
            target_module.set_input(target_input, source_signal)

        # Return the final output of the last module
        final_module_name = list(self.modules.keys())[-1]
        final_module = self.modules[final_module_name]
        return final_module.get_output(0)
    
class Mixer:
    def __init__(self, num_channels=20):
        self.num_channels = num_channels
        self.channels = [[] for _ in range(num_channels)]

    def add_to_channel(self, channel, synthesizer):
        if 0 <= channel < self.num_channels:
            self.channels[channel].append(synthesizer)
        else:
            raise ValueError(f"Invalid channel number {channel}. Must be between 0 and {self.num_channels - 1}.")

    def mix(self, normalize=True):
        num_samples = int(DURATION * SAMPLE_RATE)
        mixed_wave = np.zeros(num_samples)

        for channel in self.channels:
            for synth in channel:
                mixed_wave += synth.synthesize()

        if normalize:
            mixed_wave = self._normalize(mixed_wave)

        return mixed_wave.astype(np.int16)

    @staticmethod
    def _normalize(waveform):
        max_val = np.iinfo(np.int16).max
        min_val = np.iinfo(np.int16).min
        scale = max(max_val / np.amax(waveform), -min_val / np.amin(waveform))
        return waveform * scale
    
def setup_synths_and_mixer(fm_params, granular_params, additive_params, subtractive_params, wavetable_params, mixer_params):
    # Create synthesizer instances
    fm_synth = FMSynthesizer(**fm_params)
    granular_synth = GranularSynthesizer(**granular_params)
    additive_synth = AdditiveSynthesizer(**additive_params)
    wavetable_synth = WavetableSynthesizer(**wavetable_params)

    # Create mixer instance
    mixer = Mixer(**mixer_params)

    # Add synthesizers to mixer channels
    mixer.add_to_channel(0, fm_synth.synthesize())
    mixer.add_to_channel(1, granular_synth.synthesize())
    mixer.add_to_channel(2, additive_synth.synthesize())
    mixer.add_to_channel(4, wavetable_synth.synthesize())

    return mixer

def main():
    file_path = ''

    # Set up parameters for each synthesizer
    fm_params = {"operators": operators, "algorithm": 1, "frequency": 440, "duration": 5, "sample_rate": 44100, "amplitude": 2**15-1}
    granular_params = {"input_audio": input_audio, "grain_size": 1000, "grain_rate": 10, "duration": 5, "sample_rate": 44100, "amplitude": 2**15-1}
    additive_params = {"harmonics": [0.6, 0.3, 0.15], "frequency": 440, "duration": 5, "sample_rate": 44100, "amplitude": 2**15-1}
    subtractive_params = {"oscillator_type": "sawtooth", "frequency": 440, "duration": 5, "sample_rate": 44100, "amplitude": 2**15-1, "filter_type": "lowpass", "cutoff_frequency": 1000, "resonance": 0.5}
    wavetable_params = {"wavetables": wavetables, "frequency": 440, "duration": 5, "sample_rate": 44100, "amplitude": 2**15-1}

    # Set up mixer parameters
    mixer_params = {"num_channels": 20, "sample_rate": 44100}

    # Set up synthesizers and mixer
    mixer = setup_synths_and_mixer(fm_params, granular_params, additive_params, subtractive_params, wavetable_params, mixer_params)

    # Get the mixed output
    mixed_output = mixer.mix()
