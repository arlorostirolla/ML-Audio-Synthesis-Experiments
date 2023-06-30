import os
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
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
import random, scipy
from Model import Net, ResnetModel

def generate_data(voice, batch):
    batch, params, _ = voice.forward(batch)
    mel_spectrograms = []
    for audio in batch:
        audio = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=345)
        mel_spec = librosa.power_to_db(mel_spec)

        channel1 = mel_spec
        channel2 = scipy.ndimage.gaussian_filter(mel_spec, sigma=1) 
        channel3 = scipy.ndimage.laplace(mel_spec)

        channel1 = (channel1 - np.mean(channel1)) / np.std(channel1)
        channel2 = (channel2 - np.mean(channel2)) / np.std(channel2)
        channel3 = (channel3 - np.mean(channel3)) / np.std(channel3)
        mel_spec = np.stack([channel1, channel2, channel3])
        
        mel_spectrograms.append(mel_spec)
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

def unfreeze_next_layer(model, current_layer):
    if current_layer >= 0 and current_layer < len(model.layers):
        for param in model.layers[current_layer].parameters():
            param.requires_grad = True
    return current_layer - 1

if __name__ == "__main__":
    first_run = False
    batch_size = 32
    writer = SummaryWriter(log_dir='./Logs/Curriculum/')
    device = torch.device("cuda")
    synthconfig = SynthConfig(batch_size=batch_size)
    voice = Voice(synthconfig=synthconfig).to(device)

    if first_run:
        net = ResnetModel().to(device)
    else:
        net = load_model('./curriculum4.pth').to(device)

    current_layer = 36

    batches = 514000
    train = range(int(batches))
    val = range(900000000, 1000000000)
    
    optimizer = Adam(net.parameters(), lr=0.00001)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')  
    training_loss = 0.0
    training_acc = 0.0
    validation_loss = 0.0
    validation_acc = 0.0
    val_loss_sum = 0
    val_loss_count = 1
    moving_avg_window = 1000
    global_step = 514240

    steps_since_loss_decrease = 0
    max_num_batches = 31250000  
    min_val_loss_1000_steps = float('inf')
    unfreezing_flag = False

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

            if global_step % moving_avg_window == 0 and global_step > 0:
                moving_avg_val_loss = val_loss_sum / val_loss_count
                if moving_avg_val_loss < min_val_loss_1000_steps:
                    min_val_loss_1000_steps = moving_avg_val_loss
                    steps_since_loss_decrease = 0
                else:
                    steps_since_loss_decrease += moving_avg_window

                val_loss_sum = 0
                val_loss_count = 1
            else:
                val_loss_sum += val_loss.item()
                val_loss_count += 1

            if steps_since_loss_decrease >= 10000:
                if unfreezing_flag == False:
                    torch.save(net.state_dict(), f'curriculum4_before_unfreezing.pth')
                unfreezing_flag = True
                current_layer = unfreeze_next_layer(net, current_layer)
                print(f"\nNo improvement in validation loss for 10000 steps, unfreezing layer {current_layer}.")
                optimizer.param_groups[0]['lr'] = 0.00001
                min_val_loss_1000_steps = float('inf')
                steps_since_loss_decrease = 0
                batch_size = 32
                synthconfig = SynthConfig(batch_size=batch_size)
                voice = Voice(synthconfig=synthconfig).to(device)

            if j % 1000 == 0:
                torch.save(net.state_dict(), f'curriculum4.pth')

            if val_loss.item() < best_val_loss: 
                best_val_loss = val_loss.item()
                torch.save(net.state_dict(), f'curriculum4_best.pth')

            if j % 10 == 0 and global_step > 0:
                writer.add_scalar('Training Loss', train_loss.item(), global_step=global_step)
                writer.add_scalar('Training Accuracy', train_acc, global_step=global_step)
                writer.add_scalar('Validation Loss', val_loss.item(), global_step=global_step)
                writer.add_scalar('Validation Accuracy', val_acc, global_step=global_step)
        
            if global_step > 0:
                print('\r Epoch %d - train_size %d -Global Step %d - Training Loss: %.3f, Validation Loss: %.3f, Training Accuracy: %.3f, Validation Accuracy: %.3f' %
                    (epoch + 1, len(train), global_step, train_loss.item() , val_loss.item() , train_acc , val_acc), end='')

            if global_step % 1000 == 0 and batches < max_num_batches and global_step > 0:
                batches += 1000
                train = range(int(batches))

            global_step += 1
            file_path = "output.txt"

            with open(file_path, "w") as file:
                file.write(str(current_layer) +','+str(global_step)+','+str(batches))



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