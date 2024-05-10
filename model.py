import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TweetsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TweetClassifer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) # input dim -> 512
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim//2) # 512 -> 256
        self.fc3 = nn.Linear(self.hidden_dim//2, self.hidden_dim//4) # 256 -> 128
        self.fc4 = nn.Linear(self.hidden_dim//4, self.hidden_dim//8) # 128 -> 64
        self.fc5 = nn.Linear(self.hidden_dim//8, self.num_classes) # 64 -> num_classe

    def forward(self, x, logits=False, softmax=False):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        if logits:
            x = torch.sigmoid(self.fc5(x))
        elif softmax:
            x = torch.softmax(self.fc5(x))
        else:
            x = self.fc5(x)
        return x
    

class TweetClassiferMultiClass(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) # input dim -> 512
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim//2) # 512 -> 256
        self.fc3 = nn.Linear(self.hidden_dim//2, self.hidden_dim//4) # 256 -> 128
        self.fc4 = nn.Linear(self.hidden_dim//4, self.hidden_dim//8) # 128 -> 64
        self.fc5 = nn.Linear(self.hidden_dim//8, self.num_classes) # 64 -> num_classe

    def forward(self, x, logits=False, softmax=False):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        if logits:
            x = torch.sigmoid(self.fc5(x))
        elif softmax:
            x = torch.softmax(self.fc5(x))
        else:
            x = self.fc5(x)
        return x
    

