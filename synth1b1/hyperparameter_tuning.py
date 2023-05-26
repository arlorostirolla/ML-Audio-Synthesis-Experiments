
import torch
from torch import nn
from torchsynth.synth import Voice
from torchsynth.config import SynthConfig
from ray import tune
from Model import Net
from train import generate_data
import random
import adabound

class Trainable(tune.Trainable):
    def setup(self, config, batch_size=32):
        self.device = torch.device("cuda")
        self.synthconfig = SynthConfig(batch_size=batch_size)
        self.batches = 1000000000/batch_size

        self.train = range(int(self.batches*0.7))
        self.val = range(int(self.batches*0.7), int(self.batches*0.10))

        self.voice = Voice().to(self.device)
        self.net = Net().to(self.device)
        self.optimizer = adabound.AdaBound(self.net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], final_lr=config["final_lr"])
        self.criterion = nn.MSELoss()
        self.epochs = 5
        
    def step(self):
        running_loss = 0.0
        training_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            for j in range(0, 1000):  
                inputs, labels = generate_data(self.voice, random.choice(self.train))
                inputs = inputs.unsqueeze(1).to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                training_acc += (torch.abs(outputs - labels) < 0.1).float().mean().item()  # accuracy as proportion within 0.1 of true value

            print("Training loss: ", str(running_loss / (len(self.train) * (epoch+1))), "Training accuracy: ", str(training_acc / (len(self.train) * (epoch+1))))
            with torch.no_grad():
                for j in range(0, 1000):
                    inputs, labels = generate_data(self.voice, random.choice(self.val))
                    inputs = inputs.unsqueeze(1).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (torch.abs(outputs - labels) < 0.1).float().mean().item()  
            
            print("validation_loss: ", str(val_loss / (1000 * (epoch+1))), "validation_accuracy: ", val_acc / (1000 * (epoch+1)))

        return {"training_loss": running_loss / (len(self.train) * self.epochs), "training_accuracy": training_acc / (len(self.train) * self.epochs), 
                "validation_loss": val_loss / (len(self.val) * self.epochs), "validation_accuracy": val_acc / (len(self.val) * self.epochs)}

# replace your main function with this
if __name__ == "__main__":
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "final_lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "dropout_rate": tune.uniform(0.1, 0.5),
    }
    analysis = tune.run(
        Trainable,
        config=config,
        resources_per_trial={"cpu": 3, "gpu": 0.4},
        num_samples=1000,
    )

    print("Best config: ", analysis.get_best_config(
        metric="validation_loss", mode="min"))

    df = analysis.dataframe()
    df.to_csv("hyperparameter_tuning.csv")