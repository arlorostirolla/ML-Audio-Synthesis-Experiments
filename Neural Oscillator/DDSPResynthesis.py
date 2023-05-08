import numpy as np

from ddsp.training.models import Autoencoder
from ddsp.training import train_util
from ddsp import core, spectral_ops

import tensorflow as tf
import tensorflow_datasets as tfds
from ddsp.training import (data, decoders, encoders, models, preprocessing, 
                           train_util, trainers)
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord
import librosa, crepe, glob, os , ddsp


def prepare_dataset(audio_dir, 
                    data_dir,
                    sample_rate=16000, 
                    frame_rate=50, 
                    example_secs=4.0, 
                    hop_secs=1.0, 
                    viterbi=True, 
                    center=True):

    # Otherwise prepare new dataset locally.
    print(f'Preparing new dataset from `{audio_dir}`')

    print()
    print('Creating dataset...')
    print('This usually takes around 2-3 minutes for each minute of audio')
    print('(10 minutes of training audio -> 20-30 minutes)')
    
    audio_files = os.listdir(audio_dir)
   
    tfrecord_path_str = f'{data_dir}/train.tfrecord'

    prepare_tfrecord(input_audio_paths=audio_files, output_tfrecord_path=tfrecord_path_str, num_shards=10, \
                          sample_rate=sample_rate,frame_rate=frame_rate, example_secs=example_secs ,hop_secs=hop_secs, \
                          viterbi=viterbi, center=center)
    

    
# Load your 5-second audio recording
audio_file = './dataset/test.wav'
target_waveform, sr = librosa.load(audio_file, sr=16000, mono=True)
n_samples = target_waveform.shape
target_waveform = tf.expand_dims(target_waveform, axis=0)
dataset = prepare_dataset('C://Users//Arlo//Particle-Swarm-Synthesis//NeuralOscillator//dataset', 
                          'C://Users//Arlo//Particle-Swarm-Synthesis//NeuralOscillator')
TIME_STEPS = 1000

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

for i in range(300):
  losses = trainer.train_step(dataset_iter)
  res_str = 'step: {}\t'.format(i)
  for k, v in losses.items():
    res_str += '{}: {:.2f}\t'.format(k, v)
  print(res_str)