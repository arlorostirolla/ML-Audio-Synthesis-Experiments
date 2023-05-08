import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def make_periodic_waveform(waveform, period_length):
    waveform_length = len(waveform)
    repetitions = waveform_length // period_length
    repeated_waveform = np.tile(waveform[1*period_length:2*period_length], 100)
    return repeated_waveform

def sine(freq, amp, phase=0, length=1, rate=48000):
    length = int(length * rate)
    factor = float(freq) * (np.pi * 2) / rate
    return np.sin(np.arange(length) * factor + phase) * amp

def peak_picking(signal, threshold):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            peaks.append(i)
    return peaks

def audio_to_sine_waves(audio_file, threshold=0.1, num_harmonics=5):
    y, sr = librosa.load(audio_file, mono=False)
    #y = y[0] # take only left channel
    S = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    
    magnitude_spectrum = np.abs(S)
    
    sine_waves = []
    
    for i in range(num_harmonics):
        peak_indices = peak_picking(magnitude_spectrum[:, i], threshold)
        
        for peak_index in peak_indices:
            sine_wave = {'frequency': freqs[peak_index], 'amplitudes': np.abs(S[peak_index]), 'phases': np.angle(S[peak_index]), 'times': times}
            sine_waves.append(sine_wave)

    return sine_waves

def synthesize_sines(sine_waves):
    sines = [sine(sine_wave['frequency'], sine_wave['amplitudes'][0]/50, length=3, rate=48000, phase=sine_wave['phases'][0]) for sine_wave in sine_waves]
    sines = np.array(sines)
    sines = np.sum(sines, axis=0)
    return sines

if __name__=="__main__":
    target_file = './test.wav'
    '''num_harmonics = 10  # Number of harmonics per sine
    sine_waves = audio_to_sine_waves(target_file, threshold=-1, num_harmonics=num_harmonics)
    
    for sine_wave in sine_waves:
        print(sine_wave['frequency'], max(sine_wave['amplitudes']))
    
    sines = synthesize_sines(sine_waves)
    sd.play(sines)
    sd.wait()

    sf.write(target_file.replace('.wav', '_resynth.wav'), sines, 48000)'''
  
    normal_waveform, sr = librosa.load(target_file, mono=False)
    # Example usage

    period_length = 60000  # Replace this with the desired length of the periodic waveform

    periodic_waveform = make_periodic_waveform(normal_waveform, period_length)
    sd.play(periodic_waveform)
    sd.wait()
    # Plotting the waveforms
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(normal_waveform)
    plt.title('Normal Waveform')
    plt.subplot(2, 1, 2)
    plt.plot(periodic_waveform)
    plt.title('Periodic Waveform')
    plt.tight_layout()
    plt.show()
