import torch
import torchaudio
from torchsynth.synth import Voice
import pandas

voice = Voice()

voice = voice.to("cuda")

for i in range(100000):
    audio, params, _ = voice(i)
    for j in range(128):
        # Select synth1B1-312-6
        temp_audio = audio[j]
        temp_params = params[j]
        print(temp_params)
        #torch.save(temp_params, f'./dataset/synth1b1-{i}-{j}.pt')
        #torchaudio.save(f"./dataset/synth1B1-{i}-{j}.wav", temp_audio.unsqueeze(0).cpu(), voice.sample_rate)
