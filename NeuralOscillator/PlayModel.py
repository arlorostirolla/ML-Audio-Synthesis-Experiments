import pickle 
import ddsp
import librosa

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

# Create Processors.
harmonic = ddsp.synths.Harmonic()

noise = ddsp.synths.FilteredNoise()
add = ddsp.processors.Add(name='add')

# Create ProcessorGroup.
dag = [(harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
       (noise, ['noise_magnitudes']),
       (add, ['noise/signal', 'harmonic/signal'])]

processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                 name='processor_group')


audio, sr = librosa.load('allstar.wav', sr=16000)
outputs = model(audio)
audio_gen = model.get_audio_from_outputs(outputs)
