import numpy as np
from abc import ABC, abstractmethod
from effects_and_functions import *
import sounddevice as sd
import time
import pydub
from pydub import AudioSegment
from functools import partial
from pyswarm import pso
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch, librosa
from scipy.io.wavfile import write

SAMPLE_RATE = 44100
FREQUENCY = 440
DURATION = 5  # in seconds
AMPLITUDE = 2 ** 15 - 1
AUDIO_TYPE = "S16LE"  # 16-bit signed little-endian PCM

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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
        print(self.duration)
        print(self.sample_rate)
        t = np.linspace(0, self.duration, num_samples, False)
        modulator_indices = ((self.frequency * self.modulator_frequency_ratio * t) % 1) * len(self.modulator_wave)
        carrier_indices = ((self.frequency * t) % 1) * len(self.carrier_wave)

        modulator = np.interp(modulator_indices, np.arange(len(self.modulator_wave)), self.modulator_wave)
        carrier = np.interp(carrier_indices + self.modulator_amplitude * modulator, np.arange(len(self.carrier_wave)), self.carrier_wave)
        out = (self.amplitude * carrier).astype(np.int16)
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
        return out

class Mixer:
    def __init__(self, num_channels=3, duration=DURATION, sample_rate=SAMPLE_RATE):
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
                valid_sample_width = max(channel.dtype.itemsize, 2)  # Ensure a valid sample width
                channel_audio = AudioSegment(
                    channel.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=valid_sample_width,
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
    
    def add_effects(self, channel, effects):
        if 0 <= channel < self.num_channels:
            for effect in effects:
                self.channels[channel] = effect(self.channels[channel])
        else:
            raise ValueError(f"Invalid channel number: {channel}. Must be between 0 and {self.num_channels - 1}.")
        
def setup_synths_and_mixer(fm_params, fm_effects_params, additive_params, additive_effects_params, wavetable_params, wavetable_effects_params, mixer_params, sr, duration):
    # Create synthesizer instances
    fm_synth = FMSynthesizer(**fm_params)
    #granular_synth = GranularSynthesizer(**granular_params)
    additive_synth = AdditiveSynthesizer(**additive_params)
    wavetables = generate_wavetables(wavetable_params["harmonic_amplitudes"], wavetable_params["num_harmonics"], wavetable_params["num_wavetables"], wavetable_params["wavetable_size"])
    wavetable_synth = WavetableSynthesizer(wavetables)

    # Create mixer instance
    mixer = Mixer(duration=duration, sample_rate=sr, **mixer_params)

    # Add synthesizers to mixer channels
    mixer.add_to_channel(0, fm_synth.synthesize())
    #mixer.add_to_channel(1, granular_synth.synthesize())
    mixer.add_to_channel(1, additive_synth.synthesize())
    mixer.add_to_channel(2, wavetable_synth.synthesize())

    fm_reverb = partial(reverb, delay= fm_effects_params['delay'],decay=fm_effects_params['decay'], on = fm_effects_params['reverb_on'], sr=sr)
    wavetable_reverb = partial(reverb, delay=wavetable_effects_params['delay'] ,decay=wavetable_effects_params['decay'], on=wavetable_effects_params['reverb_on'], sr=sr)
    additive_reverb = partial(reverb, delay=additive_effects_params['delay'] ,decay=additive_effects_params['decay'], on=additive_effects_params['reverb_on'],sr=sr)

    fm_distortion = partial(distortion, threshold=fm_effects_params['distortion_threshold'], gain=fm_effects_params['distortion_gain'], on=fm_effects_params['distortion_on'])
    wavetable_distortion = partial(distortion, threshold=wavetable_effects_params['distortion_threshold'], gain=wavetable_effects_params['distortion_gain'], on=wavetable_effects_params['distortion_on'])
    additive_distortion = partial(distortion, threshold=additive_effects_params['distortion_threshold'], gain=additive_effects_params['distortion_gain'], on=additive_effects_params['distortion_on'])

    mixer.add_effects(0, [fm_distortion, fm_reverb])
    mixer.add_effects(1, [additive_distortion, additive_reverb])
    mixer.add_effects(2, [wavetable_distortion, wavetable_reverb])

    return mixer

def perceptual_loss(reference_audio, synthesized_audio):
    audio_data, sample_rate = librosa.load(reference_audio)

    synthesized_audio = synthesized_audio.astype(np.float32) / np.iinfo(synthesized_audio.dtype).max
    resampled_audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    resampled_synths = librosa.resample(synthesized_audio, orig_sr=sample_rate, target_sr=16000)  
    min_length = min(resampled_audio.shape[0], resampled_synths.shape[0])
    resampled_audio = resampled_audio[:min_length]
    resampled_synths = resampled_synths[:min_length]

    ref_input_values = processor(resampled_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
    synth_input_values = processor(resampled_synths, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)

    with torch.no_grad():
        ref_features = model(ref_input_values).last_hidden_state
        synth_features = model(synth_input_values).last_hidden_state

    loss = torch.mean((ref_features - synth_features) ** 2).item()
    print("loss:" + str(loss))
    return loss

count = 0
def objective_function(parameters):
    global count
    print("count:" + str(count))
    
    oscillator_types = ['sine', 'square', 'sawtooth', 'triangle', 'pwm', 'noise', 'sine2', 'sawtooth2', 'triangle2', 'harmonic', 'sawtooth3', 'triangle3', 'square3', 'fm_sine', 'fm_square', 'fm_sawtooth', 'fm_triangle']
    # Unpack parameters and create mixer
    # You will need to replace this with your actual parameter unpacking and mixer creation code
    pitch, duration, sample_rate, amplitude = parameters[0], parameters[1], round(parameters[2]), 2**15-1
    mfr, ma, mw, cw = parameters[3], parameters[4], parameters[5], parameters[6]
    ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9, ha10 = parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16]
    wha1, wha2, wha3, wha4, wha5, wha6, wha7, wha8, wha9, wha10 = parameters[17], parameters[18], parameters[19], parameters[20], parameters[21], parameters[22], parameters[23], parameters[24], parameters[25], parameters[26]
    num_wavetables, wavetable_size = parameters[27], parameters[28]
    wavetable_reverb_delay, wave_table_reverb_decay, fm_reverb_reverb_delay, fm_reverb_decay, additive_reverb_delay, additive_reverb_decay = parameters[29], parameters[30], parameters[31], parameters[32], parameters[33], parameters[34]
    wavetable_reverb_on, fm_reverb_on, additive_reverb_on = int(parameters[35]), int(parameters[36]), int(parameters[37])
    wavetable_distortion_threshold, wavetable_distortion_gain, fm_distortion_threshold, fm_distortion_gain, additive_distortion_threshold, additive_distortion_gain = parameters[38], parameters[39], parameters[40], parameters[41], parameters[42], parameters[43]
    wavetable_distortion_on, fm_distortion_on, additive_distortion_on = int(parameters[44]), int(parameters[45]), int(parameters[46])

    wavetable_params = {"harmonic_amplitudes": [wha1, wha2, wha3, wha4, wha5, wha6, wha7, wha8, wha9, wha10], "num_harmonics": 10,"num_wavetables": int(num_wavetables), "wavetable_size": int(wavetable_size) }
    fm_params = {"modulator_frequency_ratio": mfr, "modulator_amplitude": ma, "modulator_wave": oscillator(freq=pitch, length=duration ,osc_type=oscillator_types[int(mw)], sr=sample_rate), 
                 "carrier_wave": oscillator(freq=pitch, length=duration,osc_type=oscillator_types[int(cw)], sr=sample_rate), "sample_rate": sample_rate, "amplitude": amplitude, "duration": duration, "frequency": pitch}
    additive_params = {"harmonics_amplitudes": [ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9, ha10], "frequency": pitch, "duration": duration, "sample_rate": sample_rate, "amplitude": amplitude}
    
    wavetable_effects_params = {"delay": wavetable_reverb_delay, "decay": wave_table_reverb_decay, "reverb_on": wavetable_reverb_on, "distortion_threshold": wavetable_distortion_threshold, "distortion_gain": wavetable_distortion_gain, "distortion_on": wavetable_distortion_on}
    fm_effects_params = {"delay": fm_reverb_reverb_delay, "decay": fm_reverb_decay, "reverb_on": fm_reverb_on, "distortion_threshold": fm_distortion_threshold, "distortion_gain": fm_distortion_gain, "distortion_on": fm_distortion_on}
    additive_effects_params = {"delay": additive_reverb_delay, "decay": additive_reverb_decay, "reverb_on": additive_reverb_on, "distortion_threshold": additive_distortion_threshold, "distortion_gain": additive_distortion_gain, "distortion_on": additive_distortion_on}

    mixer = setup_synths_and_mixer(fm_params, fm_effects_params, additive_params, additive_effects_params, wavetable_params, wavetable_effects_params, {"num_channels": 3}, sample_rate, duration)

    synthesized_audio = mixer.mix()
    write(f"./outputs/output{count}.wav", sample_rate, synthesized_audio.astype(np.int16))
    loss = perceptual_loss('./528491.wav', synthesized_audio)
    count+=1
    return loss

def create_and_play_best_synth(parameters):

    oscillator_types = ['sine', 'square', 'sawtooth', 'triangle', 'pwm', 'noise', 'sine2', 'sawtooth2', 'triangle2', 'harmonic', 'sawtooth3', 'triangle3', 'square3', 'fm_sine', 'fm_square', 'fm_sawtooth', 'fm_triangle']
    pitch, duration, sample_rate, amplitude = parameters[0], parameters[1], round(parameters[2]), 2**15-1
    mfr, ma, mw, cw = parameters[3], parameters[4], parameters[5], parameters[6]
    ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9, ha10 = parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16]
    wha1, wha2, wha3, wha4, wha5, wha6, wha7, wha8, wha9, wha10 = parameters[17], parameters[18], parameters[19], parameters[20], parameters[21], parameters[22], parameters[23], parameters[24], parameters[25], parameters[26]
    num_wavetables, wavetable_size = parameters[27], parameters[28]
    wavetable_reverb_delay, wave_table_reverb_decay, fm_reverb_reverb_delay, fm_reverb_decay, additive_reverb_delay, additive_reverb_decay = parameters[29], parameters[30], parameters[31], parameters[32], parameters[33], parameters[34]
    wavetable_reverb_on, fm_reverb_on, additive_reverb_on = int(parameters[35]), int(parameters[36]), int(parameters[37])
    wavetable_distortion_threshold, wavetable_distortion_gain, fm_distortion_threshold, fm_distortion_gain, additive_distortion_threshold, additive_distortion_gain = parameters[38], parameters[39], parameters[40], parameters[41], parameters[42], parameters[43]
    wavetable_distortion_on, fm_distortion_on, additive_distortion_on = int(parameters[44]), int(parameters[45]), int(parameters[46])

    wavetable_params = {"harmonic_amplitudes": [wha1, wha2, wha3, wha4, wha5, wha6, wha7, wha8, wha9, wha10], "num_harmonics": 10,"num_wavetables": int(num_wavetables), "wavetable_size": int(wavetable_size) }
    fm_params = {"modulator_frequency_ratio": mfr, "modulator_amplitude": ma, "modulator_wave": oscillator(freq=pitch, length=duration ,osc_type=oscillator_types[int(mw)], sr=sample_rate), 
                 "carrier_wave": oscillator(freq=pitch, length=duration,osc_type=oscillator_types[int(cw)], sr=sample_rate), "sample_rate": sample_rate, "amplitude": amplitude, "duration": duration, "frequency": pitch}
    additive_params = {"harmonics_amplitudes": [ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9, ha10], "frequency": pitch, "duration": duration, "sample_rate": sample_rate, "amplitude": amplitude}
    
    wavetable_effects_params = {"delay": wavetable_reverb_delay, "decay": wave_table_reverb_decay, "reverb_on": wavetable_reverb_on, "distortion_threshold": wavetable_distortion_threshold, "distortion_gain": wavetable_distortion_gain, "distortion_on": wavetable_distortion_on}
    fm_effects_params = {"delay": fm_reverb_reverb_delay, "decay": fm_reverb_decay, "reverb_on": fm_reverb_on, "distortion_threshold": fm_distortion_threshold, "distortion_gain": fm_distortion_gain, "distortion_on": fm_distortion_on}
    additive_effects_params = {"delay": additive_reverb_delay, "decay": additive_reverb_decay, "reverb_on": additive_reverb_on, "distortion_threshold": additive_distortion_threshold, "distortion_gain": additive_distortion_gain, "distortion_on": additive_distortion_on}

    mixer = setup_synths_and_mixer(fm_params, fm_effects_params, additive_params, additive_effects_params, wavetable_params, wavetable_effects_params, {"num_channels": 3}, sample_rate, duration)

    synthesized_audio = mixer.mix()
    write("best_output.wav", sample_rate, synthesized_audio.astype(np.int16))

    return synthesized_audio


def main():
    file_path = './528491.wav'
    pitch, duration, sample_rate = analyze_wav(file_path)

    lb = [pitch, duration, sample_rate,             0.1, 0, 0,  0,  0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,   256,  0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0]
    ub = [pitch+1e-7, duration+1e-7, sample_rate+1e-7, 5.0, 1, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 2048, 2.0,  1.0, 2.0,  1.0, 2.0,  1.0, 1, 1, 1, 1.0,  2.0,  1.0,  2.0,  1.0,  2.0,  1, 1, 1]

    optimized_parameters, final_loss = pso(objective_function, lb, ub, swarmsize=460, maxiter=10000)
    create_and_play_best_synth(optimized_parameters)

if __name__ == "__main__":
    main() 
