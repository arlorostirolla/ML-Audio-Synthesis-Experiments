import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchaudio.transforms import MelSpectrogram
import torchaudio
import numpy as np
import torchsynth
from torchsynth.synth import Voice
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
from torchsynth.config import SynthConfig
import random
from Model import Net

def generate_data(voice, batch):
    # Use voice to generate synth1b1 audio samples
    batch, params, _ = voice.forward(batch)
    mel_spectrograms = []
    for audio in batch:
        audio = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spectrograms.append(mel_spec)
        
    return torch.tensor(np.array(mel_spectrograms)), params

def load_model(path):
    net_new = Net()
    net_new.load_state_dict(torch.load(path))
    return net_new

def generate(file, model_path):
    input_audio, sample_rate = torchaudio.load(file)
    input_audio = input_audio.squeeze().detach().cpu().numpy().astype(np.float32)
    mel_spec = librosa.feature.melspectrogram(y=input_audio, sr=16000, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec)

    model = load_model(model_path)
    new_params = model.forward(torch.tensor(np.array([mel_spec])))

    config = SynthConfig(reproducible=False, batch_size=1)
    voice = Voice(config)
    audio, params, _ = voice()

    for p, new_p in zip(voice.parameters(), new_params.T):
        p.data = new_p.detach()

    audio_2, params_2, _ = voice()

    return audio_2, params_2


if __name__ == "__main__":
    first_run = False
    writer = SummaryWriter(log_dir='./log3/')
    device = torch.device("cuda")
    voice = Voice().to(device)
    if first_run:
        net = Net().to(device)
    else:
        net = load_model('./net0.pth').to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    for epoch in range(10):
        running_loss = 0.0
        print_every = 1000
        global_step = 0
        for j in range(1, 7812500):
            i = random.randint(1, 7812499)
            inputs, labels = generate_data(voice, i)
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss', loss, global_step=epoch)

            if j % print_every == 0:
                torch.save(net.state_dict(), f'net{i%10}.pth')

            if j % 100 == 0:
                writer.add_scalar('Loss', loss, global_step=global_step)
            global_step += 1

            print('\r [%d, %5d] average loss: %.3f, loss: %.3f' % (epoch + 1, j + 1, running_loss / j, loss.item()), end='')