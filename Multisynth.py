import numpy as np
from abc import ABC, abstractmethod
from effects_and_functions import *
import sounddevice as sd
import time
import pydub
from pydub import AudioSegment

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

    def __init__(self, modulator_frequency_ratio, modulator_amplitude, modulator_wave, carrier_wave, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulator_frequency_ratio = modulator_frequency_ratio
        self.modulator_amplitude = modulator_amplitude
        self.modulator_wave = modulator_wave
        self.carrier_wave = carrier_wave

    def synthesize(self):
        num_samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, num_samples, False)
        modulator_indices = ((self.frequency * self.modulator_frequency_ratio * t) % 1) * len(self.modulator_wave)
        carrier_indices = ((self.frequency * t) % 1) * len(self.carrier_wave)

        modulator = np.interp(modulator_indices, np.arange(len(self.modulator_wave)), self.modulator_wave)
        carrier = np.interp(carrier_indices + self.modulator_amplitude * modulator, np.arange(len(self.carrier_wave)), self.carrier_wave)
        out = (self.amplitude * carrier).astype(np.int16)
        sd.play(out, 48000)
        time.sleep(0.5)
        sd.stop()
        return out


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
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), False)
        result = np.zeros_like(t)

        for i, amp in enumerate(self.harmonics_amplitudes):
            harmonic = (i + 1) * self.frequency
            result += amp * np.sin(2 * np.pi * harmonic * t)
        out = (self.amplitude * result).astype(np.int16)
        sd.play(out, 48000)
        time.sleep(0.5)
        sd.stop()
        return out

class WavetableSynthesizer(Synthesizer):

    def __init__(self, wavetables, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wavetables = wavetables

    def synthesize(self):
        t = np.arange(int(self.duration * self.sample_rate))
        result = np.zeros_like(t, dtype=np.float32)

        for i in range(len(t)):
            wavetable_idx = 0

            # Find the appropriate wavetable index for the current frequency
            while wavetable_idx < len(self.wavetables) - 2 and self.frequency > self.wavetables[wavetable_idx + 1][0]:
                wavetable_idx += 1

            # Get the wavetables and their corresponding frequencies
            wavetable_1_freq, wavetable_1 = self.wavetables[wavetable_idx]
            wavetable_2_freq, wavetable_2 = self.wavetables[wavetable_idx + 1]

            # Interpolate between the wavetables based on the frequency
            alpha = (self.frequency - wavetable_1_freq) / (wavetable_2_freq - wavetable_1_freq)
            result[i] = np.interp(t[i] % len(wavetable_1), np.arange(len(wavetable_1)), wavetable_1) * (1 - alpha) + np.interp(t[i] % len(wavetable_2), np.arange(len(wavetable_2)), wavetable_2) * alpha


        out = (self.amplitude * result).astype(np.int16)
        sd.play(out, 48000)
        time.sleep(0.5)
        sd.stop()
        return out


    
class Mixer:
    def __init__(self, num_channels=20, duration=DURATION, sample_rate=SAMPLE_RATE):
        self.num_channels = num_channels
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = [np.array([], dtype=np.int16) for _ in range(num_channels)]

    def add_to_channel(self, channel, synthesized_wave):
        if 0 <= channel < self.num_channels:
            synthesized_wave = np.array(synthesized_wave)

            if self.channels[channel].size == 0:  # If the channel is empty
                self.channels[channel] = synthesized_wave
            else:
                channel_wave = self.channels[channel]

                if synthesized_wave.size > channel_wave.size:
                    # Pad the channel with zeros
                    padding = np.zeros(synthesized_wave.size - channel_wave.size, dtype=synthesized_wave.dtype)
                    self.channels[channel] = np.concatenate((channel_wave, padding))

                if synthesized_wave.size < channel_wave.size:
                    # Pad the synthesized_wave with zeros
                    padding = np.zeros(channel_wave.size - synthesized_wave.size, dtype=synthesized_wave.dtype)
                    synthesized_wave = np.concatenate((synthesized_wave, padding))

                self.channels[channel] += synthesized_wave
        else:
            raise ValueError(f"Invalid channel number: {channel}. Must be between 0 and {self.num_channels - 1}.")


    def mix(self, normalize=False):
        mixed_wave = AudioSegment.silent(duration=int(self.duration * 1000), frame_rate=self.sample_rate)

        for channel in self.channels:
            if len(channel) > 0:
                channel_audio = AudioSegment(
                    channel.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=channel.dtype.itemsize,
                    channels=1
                )
                mixed_wave = mixed_wave.overlay(channel_audio)

        if normalize:
            mixed_wave = mixed_wave.normalize()

        return np.frombuffer(mixed_wave.raw_data, dtype=np.int16)



    @staticmethod
    def _normalize(waveform):
        max_val = np.iinfo(np.int16).max
        min_val = np.iinfo(np.int16).min
        scale = max(max_val / np.amax(waveform), -min_val / np.amin(waveform))
        return waveform * scale
    
def setup_synths_and_mixer(fm_params, additive_params, wavetable_params, mixer_params,sr, duration):
    # Create synthesizer instances
    fm_synth = FMSynthesizer(**fm_params)
    #granular_synth = GranularSynthesizer(**granular_params)
    additive_synth = AdditiveSynthesizer(**additive_params)
    wavetable_synth = WavetableSynthesizer(**wavetable_params)

    # Create mixer instance
    mixer = Mixer(duration=duration, sample_rate=sr, **mixer_params)

    # Add synthesizers to mixer channels
    mixer.add_to_channel(0, fm_synth.synthesize())
    #mixer.add_to_channel(1, granular_synth.synthesize())
    mixer.add_to_channel(2, additive_synth.synthesize())
    mixer.add_to_channel(4, wavetable_synth.synthesize())

    return mixer

def main():
    file_path = './528491.wav'
    pitch, duration, sample_rate = analyze_wav(file_path)
    print(pitch, duration, sample_rate)
    wavetables = generate_wavetables(1, 12, 12,256)
    # bounds: harmonic amplitudes = [0, 1], num_harmonics = [1, 50], num_wavetables = [1, 100], wavetable_size = [256, 2048]
    # bounds: grain_size = [1, 1000], grain_rate = [1, 100], overlap = [0, 1]
    #oscillator types: [0, 16]

    # Create a mixer with 5 channels

    # Set up parameters for each synthesizer
    fm_params = {"modulator_frequency_ratio": 2.0, "modulator_amplitude": 0.5, "modulator_wave": oscillator(freq=pitch, length=duration,osc_type="sine", sr=sample_rate), 
                 "carrier_wave": oscillator(freq=pitch, length=duration,osc_type="triangle", sr=sample_rate), "sample_rate": sample_rate, "amplitude": 2**15-1, "duration": duration, "frequency": pitch}
    #bounds: modulator_frequency_ratio = [0.1, 5.0], modulator_amplitude = [0, 1], modulator_waveform = [0, 16], carrier_waveform = [0, 16]

    #granular_params = {"input_audio": input_audio, "grain_size": 1000, "grain_rate": 10, "duration": duration, "sample_rate": sample_rate, "amplitude": 2**15-1}
    additive_params = {"harmonics_amplitudes": [0.6, 0.3, 0.15], "frequency": pitch, "duration": duration, "sample_rate": sample_rate, "amplitude": 2**15-1}
    #bounds: harmonics = [0, 1], num_harmonics = [1, 50]

    wavetable_params = {"wavetables": wavetables, "frequency": pitch, "duration": duration, "sample_rate": sample_rate, "amplitude": 2**15-1}

    # Set up mixer parameters
    mixer_params = {"num_channels": 20}

    # Set up synthesizers and mixer
    mixer = setup_synths_and_mixer(fm_params, additive_params, wavetable_params, mixer_params, sample_rate, duration)

    # Get the mixed output
    mixed_output = mixer.mix()
    output_file_path = 'output.wav'
    output_sample_rate = sample_rate  # You can replace this with the sample rate of your audio

    # Normalize the audio to the maximum 16-bit integer range
    mixed_output = (mixed_output / np.max(np.abs(mixed_output)) * 32767).astype(np.int16)

    # Write the audio data to a WAV file
    wavfile.write(output_file_path, output_sample_rate, mixed_output)

if __name__ == "__main__":
    main() 
