import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ScalpelMLabel(nn.Module):
    """Модель для мультилейбл-классификации с полносвязными слоями"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1))
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(x)))
        for layer in getattr(self, "hidden_layers", ()):
            x = self.dropout(self.relu(layer(x)))
        return self.fc2(x)