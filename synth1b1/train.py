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
from Model import Net, ResnetModel
import adabound
from pathos.multiprocessing import ProcessingPool as Pool



def generate_data(voice, batch):
    batch, params, _ = voice.forward(batch)
    mel_spectrograms = []
    for audio in batch:
        audio = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=345)
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spec= np.repeat(mel_spec[..., np.newaxis], 3, axis=2)
        mel_spec = mel_spec.transpose(2, 0, 1)
        spectrogram_normalized = np.zeros_like(mel_spec) 
        for i in range(3):
            spectrogram_channel = spectrogram_normalized[i, :, :]
            spectrogram_normalized[i, :, :] = librosa.pcen(spectrogram_channel + abs(spectrogram_channel.min()) + 1e-6)
        mel_spectrograms.append(spectrogram_normalized)
    return torch.tensor(np.array(mel_spectrograms)), params

def load_model(path):
    net_new = ResnetModel()
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
    batch_size = 32
    writer = SummaryWriter(log_dir='./Logs/Resnet101Run1/')
    device = torch.device("cuda")
    synthconfig = SynthConfig(batch_size=batch_size)
    voice = Voice(synthconfig=synthconfig).to(device)

    if first_run:
        net = ResnetModel().to(device)
    else:
        net = load_model('./resnet1.pth').to(device)

    batches = 1000000000/batch_size
    train = range(int(batches*0.5))
    val = range(int(batches*0.5), int(batches*0.7))
    test = range(int(batches*0.7), int(batches))

    optimizer = adabound.AdaBound(net.parameters(), lr=0.001, final_lr=0.00001)
    criterion = nn.MSELoss()
    training_loss = 0.0
    training_acc = 0.0
    validation_loss = 0.0
    validation_acc = 0.0
    save_every = 1000
    global_step = 0

    for epoch in range(1000):
        for j in range(len(train)):
            net.train()
            i = random.choice(train)
            inputs, labels = generate_data(voice, i)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            training_loss += train_loss.item()
            train_acc = (torch.abs(outputs - labels) < 0.1).float().mean().item()
            training_acc += train_acc

            net.eval() 
            with torch.no_grad():  
                i = random.choice(val)
                inputs, labels = generate_data(voice, i)
                inputs = inputs.to(device)
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)
                validation_loss += val_loss.item()
                val_acc = (torch.abs(outputs - labels) < 0.1).float().mean().item()
                validation_acc += val_acc

            if j % save_every == 0:
                torch.save(net.state_dict(), f'resnet1.pth')

            if j % 10 == 0 and global_step > 0:
                writer.add_scalar('Training Loss', train_loss.item(), global_step=global_step)
                writer.add_scalar('Training Accuracy', train_acc, global_step=global_step)
                writer.add_scalar('Validation Loss', val_loss.item(), global_step=global_step)
                writer.add_scalar('Validation Accuracy', val_acc, global_step=global_step)
            if global_step > 0:
                print('\r Epoch %d - Global Step %d - Training Loss: %.3f, Validation Loss: %.3f, Training Accuracy: %.3f, Validation Accuracy: %.3f' %
                    (epoch + 1, global_step, train_loss.item() , val_loss.item() , train_acc , val_acc), end='')

            global_step += 1

        




'''
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
            inputs = inputs.to(device)
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

            print('\r [%d, %5d] average loss: %.3f, loss: %.3f' % (epoch + 1, j + 1, running_loss / j, loss.item()), end='')'''