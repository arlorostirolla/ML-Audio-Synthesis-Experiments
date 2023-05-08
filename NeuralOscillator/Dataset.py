import os
import numpy as np
import tensorflow as tf
import pydub
from ddsp import spectral_ops
from ddsp.training.data_preparation.prepare_tfrecord_lib import  _chunk_audio, _add_f0_estimate, _add_loudness, _split_example, _float_dict_to_tfexample
from pydub import AudioSegment

CREPE_SAMPLE_RATE = spectral_ops.CREPE_SAMPLE_RATE

def _load_audio_as_array(audio_path, sample_rate):
    audio_segment = (pydub.AudioSegment.from_wav(audio_path).set_channels(1))
      # Compute expected length at given `sample_rate`
    expected_len = int(audio_segment.duration_seconds * sample_rate)
    # Resample to `sample_rate`
    audio_segment = audio_segment.set_frame_rate(sample_rate)
    sample_arr = audio_segment.get_array_of_samples()
    audio = np.array(sample_arr).astype(np.float32)
    # Zero pad missing samples, if any
    audio = spectral_ops.pad_or_trim_to_expected_length(audio, expected_len)
    # Convert from int to float representation.
    audio /= np.iinfo(sample_arr.typecode).max

    if sample_rate != CREPE_SAMPLE_RATE:
        audio_16k = _load_audio_as_array(audio_path, CREPE_SAMPLE_RATE)
    else:
        audio_16k = audio
    return {'audio': audio, 'audio_16k': audio_16k}



def process_single_audio(audio_path, output_tfrecord_path, sample_rate=16000, frame_rate=250, example_secs=4, hop_secs=1, chunk_secs=8.0, center=False, viterbi=True):
    audio_data = _load_audio_as_array(audio_path, sample_rate)
    audio_chunks = list(_chunk_audio(audio_data, sample_rate, chunk_secs))

    examples = []
    for ex in audio_chunks:
        if frame_rate:
            ex = _add_f0_estimate(ex, frame_rate, center, viterbi)
            ex = _add_loudness(ex, frame_rate, n_fft=512, center=center)

        if example_secs:
            split_examples = list(_split_example(ex, sample_rate, frame_rate, example_secs, hop_secs, center))
            examples.extend(split_examples)
        else:
            examples.append(ex)

    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        for example in examples:
            tf_example = _float_dict_to_tfexample(example)
            writer.write(tf_example.SerializeToString())
    return examples

def create_dataset_from_examples(examples, batch_size=1):
    def gen():
        for ex in examples:
            yield ex

    output_types = {k: tf.float32 for k in examples[0].keys()}
    dataset = tf.data.Dataset.from_generator(gen, output_types=output_types)
    dataset = dataset.batch(batch_size)
    return dataset


