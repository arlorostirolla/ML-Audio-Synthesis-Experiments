from Classes import *
import numpy as np
import torch
import torch
import torch.optim as optim
import torchaudio
from torchaudio.transforms import Resample

# Load the target .wav file and preprocess the audio
def load_wav_file(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0), sample_rate


if __name__ == "__main__":
    # Instantiate the synthesizer, loss function, and optimizer
    synth = CustomSynthesizer()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(synth.parameters(), lr=0.001)
    device = torch.device("cpu")
    # Load the target .wav file
    target_waveform, sample_rate = load_wav_file("./528491.wav")
    target_waveform = target_waveform.to(device)
    #target_waveform = target_waveform.squeeze(0)
    export_path = "./outputs/"
    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Generate mixed audio
        mixed_audio = synth()

        if mixed_audio.shape[-1] > target_waveform.shape[-1]:
            mixed_audio = mixed_audio[..., :target_waveform.shape[-1]]
        elif mixed_audio.shape[-1] < target_waveform.shape[-1]:
            # Pad the output audio tensor with zeros
            mixed_audio = torchaudio.transforms.PadTrim(target_waveform.shape[-1])(mixed_audio)
        
        mixed_audio = mixed_audio.squeeze(1)
        print(mixed_audio.shape)
        print(target_waveform.shape)
        mixed_audio.to(device)
        mixed_audio = mixed_audio.view(-1)
        target_waveform = target_waveform.view(-1)
        
        # Calculate the loss
        loss = criterion(mixed_audio, target_waveform)

        # Backpropagate the loss and update the parameters
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            # Convert the mixed_audio tensor back to a waveform
            mixed_waveform = torch.squeeze(mixed_audio, dim=1).cpu()
            mixed_waveform = mixed_waveform.detach().numpy()
            mixed_waveform = mixed_waveform.squeeze()
            # Export the mixed waveform as a WAV file
            torchaudio.save(export_path, mixed_waveform, sample_rate)

            print("Output waveform exported to:", export_path)