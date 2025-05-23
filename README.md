# LANCETNIC 1.2.1

[![PyPI Package Version](https://img.shields.io/pypi/v/lancetnic.svg?style=flat-square)](https://pypi.org/project/lancetnic/)
[![PyPi status](https://img.shields.io/pypi/status/lancetnic.svg?style=flat-square)](https://pypi.python.org/pypi/lancetnic)
[![Downloads](https://static.pepy.tech/badge/lancetnic)](https://pepy.tech/project/lancetnic)
[![Downloads](https://img.shields.io/pypi/dm/lancetnic.svg?style=flat-square)](https://pypi.python.org/pypi/lancetnic)
[![MIT License](https://img.shields.io/pypi/l/lancetnic.svg?style=flat-square)](https://opensource.org/licenses/MIT)

The LANCETNIC library is a tool for working with text data: learning, analysis, and inference.

Tasks to be solved:
- Binary classification (spam/not spam; patient is sick/not sick; loan approved/refusal, etc.)


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
Training:
```Python
from lancetnic.models import LancetBC
from lancetnic import Binary

model = Binary()
model.train(model_name=LancetBC, # A model for binary classification
            train_path="train.csv", # The path to the training dataset
            val_path="val.csv", # Path to the validation dataset
            num_epochs=50 # Number of training epochs
            )
            
```
Inferece:
```Python
from lancetnic import Predictor
pred=Predictor()
prediction=pred.predict(model_path="best_model.pt",
             text="Your text"
             )

print(prediction)
```