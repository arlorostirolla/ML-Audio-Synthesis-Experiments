import numpy as np

from ddsp.training.models import Autoencoder
from ddsp.training import train_util
from ddsp import core, spectral_ops
import Dataset
import tensorflow as tf
import tensorflow_datasets as tfds
from ddsp.training import (data, decoders, encoders, models, preprocessing, 
                           train_util, trainers)
from ddsp.training.data import TFRecordProvider
from ddsp.training.data_preparation.prepare_tfrecord_lib import *
import librosa, crepe, glob, os , ddsp, time
import sounddevice as sd
import soundfile as sf
import pickle

def set_shape(element):
    global n_samples
    global TIME_STEPS
    element['audio'].set_shape([1, n_samples])
    element['audio'] = tf.squeeze(element['audio'], axis=0)
    element['f0_hz'].set_shape([1, TIME_STEPS])
    element['f0_hz'] = tf.squeeze(element['f0_hz'], axis=0)
    element['f0_confidence'].set_shape([1, TIME_STEPS])
    element['f0_confidence'] = tf.squeeze(element['f0_confidence'], axis=0)
    return element

# Load your 5-second audio recording
audio_file = './dataset/test.wav'
target_waveform, sr = librosa.load(audio_file, sr=16000, mono=True)

input_audio_path = 'C:/Users/Arlo/Particle-Swarm-Synthesis/NeuralOscillator/dataset/test.wav'
output_tfrecord_path = 'C:/Users/Arlo/Particle-Swarm-Synthesis/NeuralOscillator/file.tfrecord'

examples = Dataset.process_single_audio(input_audio_path, output_tfrecord_path)
dataset = Dataset.create_dataset_from_examples(examples)
TIME_STEPS = 1000
dataset_temp = dataset.take(1).repeat()
batch = next(iter(dataset_temp))
audio = batch['audio']
n_samples = audio.shape[1]
dataset = dataset.map(set_shape)
dataset = dataset.take(1).repeat()

# Create Neural Networks.
preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=TIME_STEPS)

decoder = decoders.RnnFcDecoder(rnn_channels = 256,
                                rnn_type = 'gru',
                                ch = 256,
                                layers_per_stack = 1,
                                input_keys = ('ld_scaled', 'f0_scaled'),
                                output_splits = (('amps', 1),
                                                 ('harmonic_distribution', 45),
                                                 ('noise_magnitudes', 45)))

# Create Processors.
harmonic = ddsp.synths.Harmonic(n_samples=n_samples, 
                                sample_rate=sr,
                                name='harmonic')

noise = ddsp.synths.FilteredNoise(window_size=0,
                                  initial_bias=-10.0,
                                  name='noise')
add = ddsp.processors.Add(name='add')

# Create ProcessorGroup.
dag = [(harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
       (noise, ['noise_magnitudes']),
       (add, ['noise/signal', 'harmonic/signal'])]

processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                 name='processor_group')


# Loss_functions
spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                         mag_weight=1.0,
                                         logmag_weight=1.0)

strategy = train_util.get_strategy()
with strategy.scope():
  # Put it together in a model.
  model = models.Autoencoder(preprocessor=preprocessor,
                             encoder=None,
                             decoder=decoder,
                             processor_group=processor_group,
                             losses=[spectral_loss])
  trainer = trainers.Trainer(model, strategy, learning_rate=1e-3)


# Build model, easiest to just run forward pass.
dataset = trainer.distribute_dataset(dataset)
trainer.build(next(iter(dataset)))

dataset_iter = iter(dataset)

for i in range(500):
  losses = trainer.train_step(dataset_iter)
  res_str = 'step: {}\t'.format(i)
  for k, v in losses.items():
    res_str += '{}: {:.2f}\t'.format(k, v)
  print(res_str)

with open('model.pickle', 'wb') as handle:
  pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL) 
start_time = time.time()

controls =  model(next(dataset_iter))
audio_gen = model.get_audio_from_outputs(controls)
print(audio_gen.shape)
sd.play(np.reshape(audio_gen, (64000,)))
sd.wait()
audio_noise = controls['noise']['signal']
sd.play(np.reshape(audio_noise, (64000,)))
sd.wait()