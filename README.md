# Particle-Swarm-Synthesis

This repo contains experiments in sound synthesis. The main goal is to create a neural oscillator, that can analyse a target waveform, and create a repeating waveform that can be used as an oscillator.
Particle swarm synthesis contains experiments in optimizing an FM synth, additive synth and wavetable synth. The code runs but the output doesn't reproduce the target effectively.
Neural oscillator contains code that averages the SFTF spectrogram of the audio, and then train a DDSP model to resynthesize the waveform.
synth1b1 contains an experiment that maps the synth1b1 waveforms to the torchsynth parameters that generated them

Dependencies:
Rubberband (download cli executable here and add to path https://breakfastquay.com/rubberband/)
ddsp
tensorflow
numpy
crepe
librosa
scipy
pydub
fastdtw
sounddevice

TODO:
1. Implement DDSP resynthesis
2. Add differentiable effects models
