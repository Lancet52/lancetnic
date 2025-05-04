import torch
import torch.nn as nn

class LancetLN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LancetLN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # Первый линейный слой
        self.relu = nn.ReLU()                             # Слой активации
        self.layer2 = nn.Linear(hidden_size, num_classes) # Второй линейный слой
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out