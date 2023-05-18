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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 14 * 41, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 78)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = x.view(-1, 256 * 14 * 41)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc4(x)
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
    pass

if __name__ == "__main__":
    writer = SummaryWriter(log_dir='logs')
    device = torch.device("cuda")
    voice = Voice().to(device)
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()


    for epoch in range(10):
        running_loss = 0.0
        print_every = 100
        global_step = 0
        for i in range(1, 7812500):
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
            # Save the model and print loss every 1000 batches
            if i % print_every == 0:
                torch.save(net.state_dict(), f'net_{i}.pth')

            if i % 100 == 0:
                writer.add_scalar('Loss', loss, global_step=global_step)
            global_step += 1
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))