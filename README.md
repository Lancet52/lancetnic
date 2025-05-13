# LANCETNIC 1.1.0

[![PyPI Package Version](https://img.shields.io/pypi/v/lancetnic.svg?style=flat-square)](https://pypi.org/project/lancetnic/)
[![Downloads](https://img.shields.io/pypi/dm/lancetnic.svg?style=flat-square)](https://pypi.python.org/pypi/lancetnic)

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

### Quick start
Training:
```Python
from lancetnic.models import LancetBC
from lancetnic import Binary

model = Binary()
model.train(model_name=LancetBC,
            train_path="datasets/spam_train.csv",
            val_path="datasets/spam_val.csv",
            num_epochs=50
            )
            
```
Inferece:
```Python
from lancetnic import Predictor
pred=Predictor()
prediction=pred.predict(model_path="Your path to model (.pt)",
             text="Your text"
             )

print(prediction)
```
