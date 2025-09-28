import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lancetnic.utils import Metrics, dir
from lancetnic.engine import Trainer


# Датасет для классификиции
class ClassifierDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Классификация текстовых данных
class TextClass:
    def __init__(self, text_column='description', label_column='category', split_ratio=0.2, random_state=42):
        self.text_column = text_column
        self.label_column = label_column

        self.df_train = None
        self.df_val = None
        self.vectorizer_text = None
        self.vectorizer_scalar = None
        self.X_train = None
        self.X_val = None
        self.label_encoder = None
        self.y_train = None
        self.y_val = None
        self.input_size = None
        self.num_epochs = None
        self.num_classes = None

        self.model = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.metrics = None
        self.best_val_loss = None
        self.new_folder_path = None
        self.model_name = None
        self.train_path = None
        self.val_path = None
        self.csv_path = None
        self.split_ratio = split_ratio
        self.random_state = random_state

    # Выбор функции потерь
    def crit(self, crit_name):
        if crit_name=='CELoss':
            criterion=nn.CrossEntropyLoss()
            return criterion
        
        elif crit_name=='BCELoss':
            criterion=nn.BCELoss()
            return criterion
        
        else:
            print("Неизвестная функция потерь")
        
    # Выбор оптимизатора
    def optimaze(self, optim_name, params, lr):
        if optim_name=='Adam':
            optimizer = optim.Adam(params=params, lr=lr)
            return optimizer
        elif optim_name=='RAdam':
            optimizer = optim.RAdam(params=params, lr=lr)
            return optimizer
        elif optim_name=='SGD':
            optimizer = optim.SGD(params=params, lr=lr)
            return optimizer
        elif optim_name=='RMSprop':
            optimizer = optim.RMSprop(params=params, lr=lr)
            return optimizer
        elif optim_name=='Adadelta':
            optimizer = optim.Adadelta(params=params, lr=lr)
            return optimizer

    # Векторизация текста с отдельным валидационным набором данных   
    def vectorize_with_val_path(self):
        self.df_val = pd.read_csv(self.val_path)
        # Векторизация текста
        self.vectorizer_text = TfidfVectorizer()
        self.X_train = self.vectorizer_text.fit_transform(
            self.df_train[self.text_column]).toarray()
        self.X_val = self.vectorizer_text.transform(
            self.df_val[self.text_column]).toarray()

        # Кодирование меток
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(
            self.df_train[self.label_column])
        self.y_val = self.label_encoder.transform(
            self.df_val[self.label_column])

        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes
    
    # Векторизация текста без отдельного валидационного набора данных
    def vectorize_no_val_path(self):
        # Векторизация текста
        self.vectorizer_text = TfidfVectorizer()
        X_all = self.vectorizer_text.fit_transform(
            self.df_train[self.text_column]).toarray()

        # Кодирование меток
        self.label_encoder = LabelEncoder()
        y_all = self.label_encoder.fit_transform(
            self.df_train[self.label_column])

        # Разделение данных на тренировочную и валидационную выборки
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_all, y_all, test_size=self.split_ratio, random_state=self.random_state)

        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes

    def train(self, model_name, train_path, val_path, num_epochs, hidden_size=256, num_layers=1, batch_size=128, learning_rate=0.001, dropout=0, optim_name='Adam', crit_name='CELoss'):
        """Обучение модели с заданными параметрами и наборами данных.

        Args:
            model_name (type): Ссылка на класс модели (например, LancetMC, LancetMCA, ScalpelMC), экземпляр которой будет создан для обучения.
            train_path (str): Путь к файлу/каталогу обучающих данных.
            val_path (str): путь к файлу/каталогу проверочных данных.
            num_epochs (int): Количество эпох.
            hidden_size (int): количество нейронов в скрытых слоях. По умолчанию используется значение 256.
            num_layers (int): Количество скрытых слоев в модели. Значение по умолчанию равно 1.
            batch_size (int): количество выборок при обновлении градиента. Значение по умолчанию равно 128.
            learning_rate (float): размер шага на каждом шаге оптимизации. Значение по умолчанию равно 0,001.
            dropout (float): Коэффициент отсева для регуляризации (от 0 до 1).Значение по умолчанию равно 0.
            optim_name (str): Оптимизатор ('Adam', 'RAdam', 'SGD', 'RMSProp', 'Adadelta' и т.д.). По умолчанию используется 'Adam'.
            crit_name (str): Функция потерь ('CELoss'). По умолчанию используется значение 'CELoss'.
        """

        # Загрузка и предобработка данных
        self.model_name = model_name
        self.train_path = train_path  
        self.val_path = val_path
        self.num_epochs = num_epochs
        self.optim_name = optim_name
        self.crit_name = crit_name
        self.df_train = pd.read_csv(self.train_path)

        # Инициализация метрик
        self.mtx = Metrics()
        if val_path is None:
            try:
                self.vectorize_no_val_path()
            except Exception as e:
                print(e)
                return
        else:
            try:
                self.vectorize_with_val_path()
            except Exception as e:
                print(e)
                print("Insert true val_path")
                return

        # Настройка обучения
        self.new_folder_path = dir()

        # Создание файла для результатов
        headers = ["epoch", "train_loss", "train_acc, %",
                   "val_loss", "val_acc, %", "F1_score"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)

        # Создание DataLoader
        train_dataset = ClassifierDataset(self.X_train, self.y_train)
        val_dataset = ClassifierDataset(self.X_val, self.y_val)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Инициализация модели
        self.model = self.model_name(
            self.input_size, hidden_size, num_layers, self.num_classes, dropout)
        criterion = self.crit(crit_name=self.crit_name)
        optimizer = self.optimaze(optim_name=self.optim_name,
                                  params=self.model.parameters(),
                                  lr=learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Создание и запуск тренера
        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            label_encoder=self.label_encoder,
            vectorizer_text=self.vectorizer_text,
            vectorizer_scalar=self.vectorizer_scalar,
            new_folder_path=self.new_folder_path
        )

        # Один вызов train() и сохранение метрик
        metrics = trainer.train(
            num_epochs=num_epochs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_size=self.input_size,
            num_classes=self.num_classes,
            train_path=train_path,
            label_column=self.label_column,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optim_name=optim_name,
            crit_name=crit_name)

        # Визуализация метрик
        self.visualize_metrics(metrics)

    def visualize_metrics(self, metrics):
        """Визуализация метрик обучения"""
        self.mtx.confus_matrix(
            # Используем переданные metrics
            last_labels=metrics['all_labels'][-1],
            last_preds=metrics['all_preds'][-1],
            label_encoder=self.label_encoder.classes_,
            save_folder_path=self.new_folder_path,
            plt_name="confusion_matrix_last_model"
        )

        self.mtx.train_val_loss(
            epoch=metrics['epoch'],
            train_loss=metrics['train_loss'],
            val_loss=metrics['val_loss'],
            save_folder_path=self.new_folder_path
        )

        self.mtx.train_val_acc(
            epoch=metrics['epoch'],
            train_acc=metrics['train_acc'],
            val_acc=metrics['val_acc'],
            save_folder_path=self.new_folder_path
        )

        self.mtx.f1score(
            epoch=metrics['epoch'],
            f1_score=metrics['f1_score'],
            save_folder_path=self.new_folder_path
        )

        self.mtx.dataset_counts(
            data_path=self.train_path,
            label_column=self.label_column,
            save_folder_path=self.new_folder_path
        )

    def predict(self, model_path, text):
        """Инференс модели

        Args:
            model_path (str): Путь до модели
            text (str): Текстовые данные
        """
        self.model_path=f"{model_path}"
        self.text=text
        # Загружаем на CPU. Так как векторизация в базовом трейне была через библиотеку sklearn, то только CPU!!!Пока так. 
        self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)  
        self.model = self.checkpoint['model'] 
        self.model.eval()        
        
        X = self.checkpoint['vectorizer_text'].transform([self.text]).toarray()
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            self.pred = torch.argmax(self.model(X), dim=1).item()
            self.class_name = self.checkpoint['label_encoder'].inverse_transform([self.pred])[0]

        return self.class_name
    
# Классификация текстовых и числовых данных    
class TextScalarClass:
    def __init__(self, text_column=None, data_column=None, label_column=None, split_ratio=0.2, random_state=42):
        self.text_column = text_column
        self.data_column = data_column
        self.label_column = label_column

        self.df_train = None
        self.df_val = None
        self.vectorizer_text = None
        self.vectorizer_scalar = None
        self.X_train = None
        self.X_val = None
        self.label_encoder = None
        self.y_train = None
        self.y_val = None
        self.input_size = None
        self.num_epochs = None
        self.num_classes = None

        self.model = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.metrics = None
        self.best_val_loss = None
        self.new_folder_path = None
        self.model_name = None
        self.train_path = None
        self.val_path = None
        self.csv_path = None
        self.split_ratio = split_ratio
        self.random_state = random_state

    # Выбор функции потерь
    def crit(self, crit_name):
        if crit_name=='CELoss':
            criterion=nn.CrossEntropyLoss()
            return criterion
        
        elif crit_name=='BCELoss':
            criterion=nn.BCELoss()
            return criterion
        
    # Выбор оптимизатора
    def optimaze(self, optim_name, params, lr):
        if optim_name=='Adam':
            optimizer = optim.Adam(params=params, lr=lr)
            return optimizer
        elif optim_name=='RAdam':
            optimizer = optim.RAdam(params=params, lr=lr)
            return optimizer
        elif optim_name=='SGD':
            optimizer = optim.SGD(params=params, lr=lr)
            return optimizer
        elif optim_name=='RMSprop':
            optimizer = optim.RMSprop(params=params, lr=lr)
            return optimizer
        elif optim_name=='Adadelta':
            optimizer = optim.Adadelta(params=params, lr=lr)
            return optimizer
        
    # Векторизация с отдельным валидационным набором данных 
    def vectorize_with_val_path(self):
        # Чтение валидационного датасета
        self.df_val = pd.read_csv(self.val_path)
        # Векторизация массива текстовых и числовых данных
        if self.text_column==None:
            # Векторизация числовых признаков для train
            self.vectorizer_scalar = StandardScaler()
            
            self.X_train = self.vectorizer_scalar.fit_transform(
                self.df_train[self.data_column].values)
            # Векторизация числовых признаков для val
            self.X_val = self.vectorizer_scalar.transform(
                self.df_val[self.data_column].values)
            
            
        else:
            
            # Векторизация текстовых признаков для train
            self.vectorizer_text = TfidfVectorizer()
            self.text_encoder = self.vectorizer_text.fit_transform(
                self.df_train[self.text_column]).toarray()
            

            # Векторизация числовых признаков для train
            self.vectorizer_scalar = StandardScaler()
            self.scalar_encoder = self.vectorizer_scalar.fit_transform(
                self.df_train[self.data_column].values)

            # Объединение текстовых и числовых признаков для train
            self.X_train = np.hstack([self.text_encoder, self.scalar_encoder])


            # Векторизация текстовых признаков для val            
            self.text_encoder_val = self.vectorizer_text.fit_transform(
                self.df_val[self.text_column]).toarray()
            

            # Векторизация числовых признаков для val            
            self.scalar_encoder_val = self.vectorizer_scalar.transform(
                self.df_val[self.data_column].values)

            # Объединение тикера и числовых признаков для val
            self.X_val = np.hstack([self.text_encoder_val, self.scalar_encoder_val])

        
        # Кодирование меток
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(
            self.df_train[self.label_column])
        self.y_val = self.label_encoder.transform(
            self.df_val[self.label_column])

        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes

    # Векторизация без отдельного набора данных
    def vectorize_no_val_path(self):
        
        # Векторизация массива текстовых и числовых данных
        if self.text_column==None:
            # Векторизация числовых признаков
            self.vectorizer_scalar = StandardScaler()
            X_all = self.vectorizer_scalar.fit_transform(
                self.df_train[self.data_column].values)
            
                       
        else:
            # Векторизация текста
            self.vectorizer_text = TfidfVectorizer()
            self.text_encoder = self.vectorizer_text.fit_transform(
                self.df_train[self.text_column]).toarray()
            

            # Векторизация числовых признаков
            self.vectorizer_scalar = StandardScaler()
            self.scalar_encoder = self.vectorizer_scalar.fit_transform(
                self.df_train[self.data_column].values)

            # Объединение тикера и числовых признаков
            X_all = np.hstack([self.text_encoder, self.scalar_encoder])

        # Кодирование меток
        self.label_encoder = LabelEncoder()
        y_all = self.label_encoder.fit_transform(
                self.df_train[self.label_column])
        

        # Разделение данных на обучающую и валидационную выборку
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_all,
                                                                              y_all,
                                                                              test_size=self.split_ratio,
                                                                              random_state=self.random_state)

        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)

        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes
    
    def train(self, model_name, train_path, val_path, num_epochs, hidden_size=256, num_layers=1, batch_size=128, learning_rate=0.001, dropout=0, optim_name='Adam', crit_name='CELoss'):
        """Обучение модели с заданными параметрами и наборами данных.

        Args:
            model_name (type): Ссылка на класс модели (например, LancetMC, LancetMCA, ScalpelMC), экземпляр которой будет создан для обучения.
            train_path (str): Путь к файлу/каталогу обучающих данных.
            val_path (str): путь к файлу/каталогу проверочных данных.
            num_epochs (int): Количество эпох.
            hidden_size (int): количество нейронов в скрытых слоях. По умолчанию используется значение 256.
            num_layers (int): Количество скрытых слоев в модели. Значение по умолчанию равно 1.
            batch_size (int): количество выборок при обновлении градиента. Значение по умолчанию равно 128.
            learning_rate (float): размер шага на каждом шаге оптимизации. Значение по умолчанию равно 0,001.
            dropout (float): Коэффициент отсева для регуляризации (от 0 до 1).Значение по умолчанию равно 0.
            optim_name (str): Оптимизатор ('Adam', 'RAdam', 'SGD', 'RMSProp', 'Adadelta' и т.д.). По умолчанию используется 'Adam'.
            crit_name (str): Функция потерь ('CELoss'). По умолчанию используется значение 'CELoss'.
        """
        # Загрузка и предобработка данных
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epochs = num_epochs
        self.optim_name = optim_name
        self.crit_name = crit_name
        self.df_train = pd.read_csv(self.train_path)

        # Инициализация метрик
        self.mtx = Metrics()
        if val_path is None:
            try:
                self.vectorize_no_val_path()
            except Exception as e:
                print(e)
                return
        else:
            try:
                self.vectorize_with_val_path()
            except Exception as e:
                print(e)
                print("Insert true val_path")
                return

        # Настройка обучения
        self.new_folder_path = dir()

        # Создание файла для результатов
        headers = ["epoch", "train_loss", "train_acc, %",
                   "val_loss", "val_acc, %", "F1_score"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)

        # Создание DataLoader
        train_dataset = ClassifierDataset(self.X_train, self.y_train)
        val_dataset = ClassifierDataset(self.X_val, self.y_val)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Инициализация модели
        self.model = self.model_name(
            self.input_size, hidden_size, num_layers, self.num_classes, dropout)
        criterion = self.crit(crit_name=self.crit_name)
        optimizer = self.optimaze(optim_name=self.optim_name,
                                  params=self.model.parameters(),
                                  lr=learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Создание и запуск тренера
        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            label_encoder=self.label_encoder,
            vectorizer_text=self.vectorizer_text,
            vectorizer_scalar=self.vectorizer_scalar,
            new_folder_path=self.new_folder_path
        )

        # Один вызов train() и сохранение метрик
        metrics = trainer.train(
            num_epochs=num_epochs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_size=self.input_size,
            num_classes=self.num_classes,
            train_path=train_path,
            label_column=self.label_column,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optim_name=optim_name,
            crit_name=crit_name)

        # Визуализация метрик
        self.visualize_metrics(metrics)

    def visualize_metrics(self, metrics):
        # Визуализация метрик обучения
        self.mtx.confus_matrix(
            # Используем переданные metrics
            last_labels=metrics['all_labels'][-1],
            last_preds=metrics['all_preds'][-1],
            label_encoder=self.label_encoder.classes_,
            save_folder_path=self.new_folder_path,
            plt_name="confusion_matrix_last_model"
        )

        self.mtx.train_val_loss(
            epoch=metrics['epoch'],
            train_loss=metrics['train_loss'],
            val_loss=metrics['val_loss'],
            save_folder_path=self.new_folder_path
        )

        self.mtx.train_val_acc(
            epoch=metrics['epoch'],
            train_acc=metrics['train_acc'],
            val_acc=metrics['val_acc'],
            save_folder_path=self.new_folder_path
        )

        self.mtx.f1score(
            epoch=metrics['epoch'],
            f1_score=metrics['f1_score'],
            save_folder_path=self.new_folder_path
        )

        self.mtx.dataset_counts(
            data_path=self.train_path,
            label_column=self.label_column,
            save_folder_path=self.new_folder_path
        )
    def predict(self, model_path, text, numeric):
        """Инференс модели

        Args:
            model_path (str): Путь до модели
            text (str): Текстовые данные
            numeric (list): Числовые данные
        """
        self.model_path=f"{model_path}"
        self.text=text
        self.numeric=numeric
        # Загружаем на CPU. Так как векторизация в базовом трейне была через библиотеку sklearn, то только CPU!!!
        self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)  
        self.model = self.checkpoint['model'] 
        self.model.eval()  

        
        if self.text==None:
            X=self.checkpoint['vectorizer_scalar'].transform([self.numeric])
        else:

            X_text = self.checkpoint['vectorizer_text'].transform([self.text]).toarray()
            X_data=self.checkpoint['vectorizer_scalar'].transform([self.numeric])
            X = np.hstack([X_text, X_data])
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            self.pred = torch.argmax(self.model(X), dim=1).item()
            self.class_name = self.checkpoint['label_encoder'].inverse_transform([self.pred])[0]

        return self.class_name
