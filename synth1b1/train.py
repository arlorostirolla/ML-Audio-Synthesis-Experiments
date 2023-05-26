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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 1024, 3)
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1024* 2* 8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 78)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(-1, 1024* 2* 8)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = F.sigmoid(self.fc6(x))
        return x


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

def generate(file):
    input_audio, sample_rate = torchaudio.load(file)
    # We set reproducible False so we can call Voice without passing in a batch_num
    config = SynthConfig(reproducible=False, batch_size=1)
    voice = Voice(config)
    audio, params, _ = voice()

    # (batch_size, num_params)
    assert params.shape == (1, 78)

    # Generate new random params in range [0,1], which is required for internal
    # torchsynth parameters
    new_params = torch.rand_like(params)

    # Now directly set Voice params
    for p, new_p in zip(voice.parameters(), new_params.T):
        p.data = new_p.detach()

    audio_2, params_2, _ = voice()

    #   output of voice is different now and used updated params
    assert not torch.allclose(audio_2, audio)
    assert torch.allclose(params_2, new_params)
    pass



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

            if i % print_every == 0:
                torch.save(net.state_dict(), f'net{i%10}.pth')

            if i % 100 == 0:
                writer.add_scalar('Loss', loss, global_step=global_step)
            global_step += 1

            print('\r [%d, %5d] average loss: %.3f, loss: %.3f' % (epoch + 1, j + 1, running_loss / j, loss.item()), end='')