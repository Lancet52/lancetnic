import torch
import torch.nn as nn
import torch.nn.functional as F

# Модель для регрессии с полносвязными слоями
class ScalpelReg(nn.Module):
    """Модель для регрессии с полносвязными слоями"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(ScalpelReg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
                
        for _ in range(self.num_layers-1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Выходной слой для регрессии (один нейрон)
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

# Модель для регрессии с LSTM
# Модель для регрессии с LSTM
class LancetReg(nn.Module):
    """Модель для регрессии с LSTM-слоем"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)