# LANCETNIC 

Библиотека LANCETNIC представляет собой инструмент для работы с текстовыми данными: обучение, анализ, инференс.

Решаемые задачи:
- Бинарная классификация (спам/не спам; болен пациент/не болен; одобрен кредит/отказ и др.)


## 🔑 Версии библиотек. Требования:

- Python==3.10.9
- torch==2.5.1+cu124
- torchaudio==2.5.1+cu124
- torchvision==0.20.1+cu124
- scikit-learn==1.6.1
- pandas==2.2.3
- matplotlib
- seaborn



## 🚀 Установка:

- Скачать репозиторий



## 👥 Авторы

- [Сазонов Антон](https://github.com/Lancet52)


## 📄 Документация
### Быстрый старт
```
from models.lancet_binary import LancetBC
from engine.trainer import Binary

model = Binary()
model.train(model_name=LancetBC,
            train_path="datasets/spam_train.csv",
            val_path="datasets/spam_val.csv",
            num_epochs=50
            )
```