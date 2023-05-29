
import torch
from torch import nn
from torchsynth.synth import Voice
from torchsynth.config import SynthConfig
from ray import air, tune
from ray.air import session
from Model import Net
from train import generate_data
import random
import adabound
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
import optuna
from ray.tune.search.optuna import OptunaSearch

class Trainable(tune.Trainable):
    def setup(self, config, batch_size=32):
        self.device = torch.device("cuda")
        self.synthconfig = SynthConfig(batch_size=batch_size)
        self.batches = 1000000000/batch_size

        self.train = range(int(self.batches*0.7))
        self.val = range(int(self.batches*0.7), int(self.batches))

        self.voice = Voice().to(self.device)
        self.net = Net().to(self.device)
        self.optimizer = adabound.AdaBound(self.net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], final_lr=config["final_lr"])
        self.criterion = nn.MSELoss()
        self.training_loss = 0.0
        self.training_acc = 0.0
        self.steps = 0  

    def step(self):
        inputs, labels = generate_data(self.voice, random.choice(self.train))
        inputs = inputs.unsqueeze(1).to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.training_loss += loss.item()
        self.training_acc += (torch.abs(outputs - labels) < 0.1).float().mean().item()
        self.steps += 1
        print("Training loss: ", str(self.training_loss / self.steps)), "Training accuracy: ", str(self.training_acc / self.steps)
        return {"training_loss": self.training_loss / self.steps+1, "training_accuracy": self.training_acc / self.steps}

if __name__ == "__main__":
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "final_lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "dropout_rate": tune.uniform(0.1, 0.5),
    }
    search_alg = OptunaSearch(metric="training_loss", mode="min", space=config)
    
    analysis = tune.run(
        Trainable,
        search_alg=search_alg,
        resources_per_trial={"cpu": 5, "gpu": 1},
        num_samples=1000,
        scheduler=ASHAScheduler(metric="training_loss", mode="max")
    )

    print("Best config: ", analysis.get_best_config(
        metric="validation_loss", mode="min"))

    df = analysis.dataframe()
    df.to_csv("hyperparameter_tuning.csv")