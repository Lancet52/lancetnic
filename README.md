# LANCETNIC 2.0.2

[![PyPI Package Version](https://img.shields.io/pypi/v/lancetnic.svg?style=flat-square)](https://pypi.org/project/lancetnic/)
[![PyPi status](https://img.shields.io/pypi/status/lancetnic.svg?style=flat-square)](https://pypi.python.org/pypi/lancetnic)
[![Downloads](https://static.pepy.tech/badge/lancetnic)](https://pepy.tech/project/lancetnic)
[![Downloads](https://img.shields.io/pypi/dm/lancetnic.svg?style=flat-square)](https://pypi.python.org/pypi/lancetnic)
[![MIT License](https://img.shields.io/pypi/l/lancetnic.svg?style=flat-square)](https://opensource.org/licenses/MIT)

LANCETNIC is a library with built-in neural network models for working with text and numeric data. Lancetnic provides convenient tools for:

- Data preparation and vectorization
- Training classification models
- Visualization of metrics
- Forecasting on new data

The library allows you to work with purely textual data, as well as with a combination of textual and numerical features, trend and price analysis.
Usage examples: text classification, identification of spam, fraudulent messages, working with numerical series and time signs.

## 🚀 Installing:
Install with CUDA

To work with the GPU, it is recommended to install PyTorch with CUDA support (OPTIONAL):

```bash
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Then install lancetnic:

```bash
pip install lancetnic
```

## 👥 Autors

- [Lancet52](https://github.com/Lancet52)

## 📄 Documentation

### [Документация на русском](https://github.com/Lancet52/lancetnic/blob/main/lancetnic/docs/RU.md)
### [Documentation in English](https://github.com/Lancet52/lancetnic/blob/main/lancetnic/docs/EN.md)


## Quick start
### Text classification example
```Python
from lancetnic.models import LancetMC
from lancetnic import TextClass

text_model = TextClass(
            text_column='description',  # Column name containing text data
            label_column='category',    # Column name containing labels
            split_ratio=0.2,            # Train/validation split ratio (if no val_path)
            random_state=42             # Random seed for reproducibility
            )

text_model.train(model_name=LancetMC,   # Model architecture for text classification
                train_path="train.csv", # Path to training data (CSV format)
                val_path="val.csv",     # Path to validation data (None for auto-split)
                num_epochs=50,          # Total training epochs
                hidden_size=256,        # Size of hidden layers
                num_layers=1,           # Number of hidden layers
                batch_size=256,         # Batch size for training
                learning_rate=0.001,    # Learning rate for optimizer
                dropout=0,              # Dropout rate (0-1)
                optim_name='Adam',      # Optimizer ('Adam', 'SGD', 'RAdam', etc.)
                crit_name='CELoss'      # Loss function ('CELoss' or 'BCELoss')
                )
           
```
### Making predictions
```Python
from lancetnic import TextClass

text_model = TextClass()
text_pred = text_model.predict(
                model_path="model.pth", # Path to saved model
                text="Sample text to classify" # Text input for prediction
                )
```

### Combined text and numeric features example
```Python
from lancetnic.models import LancetMC
from lancetnic import TextScalarClass

mixed_model = TextScalarClass(
                text_column='description',  # Text column name (None if only numeric)
                data_column=['feat1', 'feat2'], # List of numeric feature columns
                label_column='target',     # Target variable column
                split_ratio=0.2,            # Train/val split ratio
                random_state=42             # Random seed
                )

mixed_model.train(model_name=LancetMC,   # Model architecture for text classification
                train_path="train.csv", # Path to training data (CSV format)
                val_path="val.csv",     # Path to validation data (None for auto-split)
                num_epochs=50,          # Total training epochs
                hidden_size=256,        # Size of hidden layers
                num_layers=1,           # Number of hidden layers
                batch_size=256,         # Batch size for training
                learning_rate=0.001,    # Learning rate for optimizer
                dropout=0,              # Dropout rate (0-1)
                optim_name='Adam',      # Optimizer ('Adam', 'SGD', 'RAdam', etc.)
                crit_name='CELoss'      # Loss function ('CELoss' or 'BCELoss')
                )
```

### Making predictions
```Python
from lancetnic import TextScalarClass

mixed_model = TextScalarClass()
mixed_pred = mixed_model.predict(
                model_path="mixed_model.pth", # Path to saved model
                text="Product description",  # Text input (None if only numeric)
                numeric=[0.5, 1.2]            # Numeric features as list
                )
```


## There are two classes of basic models in LANCETNIC: 

- LancetMC
```Python
from lancetnic.models import LancetMC
```
- LancetMCA
```Python
from lancetnic.models import LancetMC
```


| Key Differences Between Models | LancetMC          | LancetMCA                     |
|--------------------------------|-------------------|-------------------------------|
| Feature                        |                   |                               |
| Core Architecture              | Basic LSTM        | LSTM + Attention              |
| Complexity                     | Lower             | Higher                        |
| Computational Cost             | Less resource-intensive | More resource-intensive  |
| Best For                       | Pure text classification | Mixed data or complex patterns |
| Interpretability               | Standard          | Provides attention weights    |
| Sequence Handling              | Good              | Excellent for long sequences  |


