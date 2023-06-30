import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import laion_clap


class Net(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 1024, 3)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(1024* 8* 8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 78)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
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

class ResnetModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ResnetModel, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        num_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        #for param in self.resnet.parameters():
        #    param.requires_grad = False

        self.layers = []
        for module in self.resnet.children():
            if isinstance(module, nn.Sequential):
                for submodule in module.children():
                    if isinstance(submodule, nn.Sequential):
                        for layer in submodule.children():
                            self.layers.append(layer)
                    else:
                        self.layers.append(submodule)
            else:
                self.layers.append(module)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.dropout5 = nn.Dropout(dropout_rate)

        # Define the new output module
        self.fc = nn.Sequential(
            nn.Linear(num_in_features, 2048),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 78), 
            nn.Sigmoid()
            ,
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
device = torch.device("cuda:0")
    
class ClapModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ClapModel, self).__init__()
        
        self.model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
        self.model.load_ckpt() 

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.dropout5 = nn.Dropout(dropout_rate)

        # Define the new output module
        self.fc = nn.Sequential(
            nn.Linear(5, 2048),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 78), 
            nn.Sigmoid()
            ,
        )

    def forward(self, x):
        x = self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
        print(x)
        print(x.shape)
        x = self.fc(x)
        return x




   