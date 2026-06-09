import torch.nn as nn

class ScalpelMTSK(nn.Module):
    """Мультимодель объединяющая классификацию и регрессию"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(ScalpelMTSK, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
            
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
        self.class_head = nn.Linear(hidden_size, num_classes)
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.model(x)
        class_out = self.class_head(features)
        reg_out = self.reg_head(features)
        return class_out, reg_out