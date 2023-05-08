import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
import sounddevice as sd
from scipy.ndimage import gaussian_filter
import numpy as np
import soundfile as sf
from fastdtw import fastdtw
from pydub import AudioSegment
from scipy.fftpack import fft, ifft
from scipy.signal import resample
from scipy import signal
import pyrubberband as pyrb
import subprocess

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

def find_best_loop_point(audio, sr, search_duration=5):
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
    return tiled_output

def pitch_shift_wav(input_file, output_folder):
    # Load the .wav file
    y, sr = librosa.load(input_file, sr=None)
    
    # Estimate the pitch of the input audio
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_index = np.argmax(magnitudes, axis=0)
    pitch_values = pitches[pitch_index, np.arange(pitches.shape[1])]
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) > 0:
        original_pitch = np.median(pitch_values)
    else:
        original_pitch = 0

    # Convert the original pitch to MIDI
    original_midi = librosa.hz_to_midi(original_pitch)

    # Rubberband executable path
    rubberband_executable = "rubberband"
    min_midi=int(original_midi)-5
    max_midi=int(original_midi)+5
    for target_midi in range(min_midi, max_midi + 1):
        # Compute the pitch shifting factor
        pitch_shift_semitones = target_midi - original_midi

        # Create the output file path
        output_file = f"{output_folder}/output_midi_{target_midi}.wav"

        # Build the Rubberband command
        command = [
            rubberband_executable,
            "-p", str(pitch_shift_semitones),
            input_file,
            output_file
        ]

        # Run the Rubberband command
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




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
    looped_audio = create_seamless_loop(audio, 5, sr)  # Change 5 to the desired number of repetitions
    sf.write('test.wav', looped_audio, sr)
    sd.play(looped_audio)
    sd.wait()
    pitch_shift_wav('./test.wav', './pitches/')

if __name__ == "__main__":
    main()
    

