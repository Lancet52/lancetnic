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

## Структура проекта

lancetnic/
├── lancetnic/
│   ├── __init__.py
│   ├── engine/
│   │   ├── trainer.py                  # Универсальный Trainer
│   │   ├── validator.py                # Валидация моделей
│   │   └── predictor.py                # Предсказания
│   │
│   ├── tasks/                          # Поддержка разных задач
│   │   ├── binary_classification.py
│   │   ├── multi_classification.py
│   │   └── __init__.py          
│   │
│   ├── models/                         # Скрытые архитектуры
│   │   ├── binary_model.py             # Для бинарной классификации
│   │   ├── multi_model.py              # Для многоклассовой
│   │   └── __init__.py                 # Инкапсуляция загрузки
│   │
│   ├── data/                           # Обработка данных
│   │   ├── tokenizer.py                # Токенизация текста
│   │   └── dataset.py                  # Dataset для PyTorch
│   │
│   └── utils/
│       ├── logger.py
│       └── metrics.py                   # Accuracy, F1, ROC-AUC...
│
├── configs/                             # Конфиги для задач
│   ├── binary.yaml
│   └── multi_class.yaml
│
├── examples/
│   ├── train_binary.py
│   └── train_multi.py
└── ...

## 🚀 Установка:

- Скачать репозиторий



## 👥 Авторы

- [Сазонов Антон](https://github.com/Lancet52)


## 📄 Документация
### Быстрый старт
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

