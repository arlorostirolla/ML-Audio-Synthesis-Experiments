import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from scipy.ndimage import gaussian_filter
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from pydub import playback
import mido, time
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def horizontal_average(mel_spec):
    return np.mean(mel_spec, axis=1)

def smooth_spectrogram(spec, sigma=1):
    return gaussian_filter(spec, sigma=sigma)

def create_avg_mel_spectrogram(avg_amplitude, mel_spec):
    avg_mel_spec = np.tile(avg_amplitude, (mel_spec.shape[1], 1)).T
    return avg_mel_spec

def mel_to_audio_griffin_lim(mel_spec, sr, n_fft, hop_length, n_iter=100):
    stft_spec = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, n_fft=n_fft)
    y_inv = librosa.griffinlim(stft_spec, hop_length=hop_length, n_iter=n_iter, window='hann')
    return y_inv

def generate_variable_length_audio(hor_avg, D_mag, D, output_length, pitch_shift, sr):
    avg_spec = np.tile(hor_avg, (output_length, 1)).T
    avg_spec = smooth_spectrogram(avg_spec, sigma=1)
    
    # Apply pitch shift
    avg_spec_shifted = librosa.effects.pitch_shift(avg_spec.T, sr=sr, n_steps=pitch_shift).T
    
    # Reconstruct the signal
    D_avg = avg_spec_shifted * np.exp(1j * np.angle(D))
    y_inv = librosa.istft(D_avg, hop_length=512, window='hann')
    
    #sd.play(y_inv)
    #sd.wait()
    return y_inv


def find_best_loop_point(audio, sr, search_duration=20):
    search_samples = int(search_duration * sr)
    search_window = audio[:search_samples]
    search_target = audio[-search_samples:]
    _, path = fastdtw(search_window, search_target)
    best_loop_point = path[-1][0]
    return best_loop_point

def create_seamless_loop(audio, num_repetitions, sr):
    # Ensure audio is one-dimensional
    audio = np.squeeze(audio)
    audio = audio.astype(float)
    
    # Find the best loop point
    loop_point = find_best_loop_point(audio, sr)
    
    # Extract the loopable section
    loopable_section = audio[:loop_point]
    sf.write('./test_loop.wav', loopable_section, sr)
    # Tile the loopable section
    tiled_output = np.tile(loopable_section, num_repetitions-1)
    
    sd.play(tiled_output)
    sd.wait()
    return tiled_output

def main():
    # Load audio file
    target_audiofile = '528491.wav'
    y, sr = librosa.load(target_audiofile, sr=None)
    y = y / np.max(np.abs(y))
    # Compute STFT
    D = librosa.stft(y, n_fft=40000, hop_length=512, window='hann')
    D_mag = np.abs(D)
    # Compute horizontal average
    hor_avg = horizontal_average(D_mag)
    audio = generate_variable_length_audio(hor_avg, D_mag, D, 480, 0, sr)
    audio = audio[int(sr*0.5):]
    looped_audio = create_seamless_loop(audio, 20, sr)  # Change 5 to the desired number of repetitions
   

    sf.write('test.wav', looped_audio, sr)

if __name__ == "__main__":
    main()
    

