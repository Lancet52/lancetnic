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
from lancetnic.engine import Trainer, RegressionTrainer
from lancetnic.engine.vectorizer import vectorize_text, vectorize_data


# Датасет для классификиции
class ClassifierDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Датасет для регрессии
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Классификация текстовых и числовых данных    
class Classification:
    def __init__(self, text_column=None, data_column=None, label_column=None, split_ratio=0.2, random_state=42, max_features=None):
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
        self.max_features = max_features
        
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
        
    # Векторизация с отдельным валидационным набором данных (как я заебался этот модуль писать....)
    def vectorize_with_val_path(self):
        # Чтение валидационного датасета
        self.df_val = pd.read_csv(self.val_path)
        
        # Векторизация только числовых данных, при отсутствии текстовых
        if self.text_column is None and self.data_column is not None:            
            if isinstance(self.data_column, str):
                self.data_column = [self.data_column]
            
            self.vectorizer_scalar = []
            self.data_encoder_list_train = []
            self.data_encoder_list_val = []
            
            for data_col in self.data_column:
                vectorizer_data = StandardScaler()                
                # Обработка train данных
                data_encoder_train = vectorizer_data.fit_transform(self.df_train[data_col].fillna(0).values.reshape(-1, 1))
                # Обработка val данных
                data_encoder_val = vectorizer_data.transform(self.df_val[data_col].fillna(0).values.reshape(-1, 1))
                
                self.data_encoder_list_train.append(data_encoder_train)
                self.data_encoder_list_val.append(data_encoder_val)
                self.vectorizer_scalar.append(vectorizer_data)
            
            self.X_train = np.hstack(self.data_encoder_list_train)
            self.X_val = np.hstack(self.data_encoder_list_val)
        
        # Векторизация только текстовых данных, при отсутсвии числовых
        elif self.data_column is None and self.text_column is not None:
            if isinstance(self.text_column, str):
                self.text_column = [self.text_column]
            
            self.vectorizer_text = []
            self.text_encoder_list_train = []
            self.text_encoder_list_val = []
            
            for text_col in self.text_column:
                vectorizer_text = TfidfVectorizer(max_features=self.max_features)
                
                # Обработка train данных
                text_encoder_train = vectorizer_text.fit_transform(self.df_train[text_col].fillna('').astype(str)).toarray()
                # Обработка val данных
                text_encoder_val = vectorizer_text.transform(self.df_val[text_col].fillna('').astype(str)).toarray()
                
                self.text_encoder_list_train.append(text_encoder_train)
                self.text_encoder_list_val.append(text_encoder_val)
                self.vectorizer_text.append(vectorizer_text)
            
            self.X_train = np.hstack(self.text_encoder_list_train)
            self.X_val = np.hstack(self.text_encoder_list_val)

        # Векторизация комбинированных (текстовых и числовых) данных     
        elif self.text_column is not None and self.data_column is not None:
            if isinstance(self.text_column, str):
                self.text_column = [self.text_column]
            if isinstance(self.data_column, str):
                self.data_column = [self.data_column]
            
            # Векторизация текстовых данных
            self.vectorizer_text = []
            self.text_encoder_list_train = []
            self.text_encoder_list_val = []
            
            for text_col in self.text_column:
                vectorizer_text = TfidfVectorizer(max_features=self.max_features)
                
                text_encoder_train = vectorizer_text.fit_transform(self.df_train[text_col].fillna('').astype(str)).toarray()
                text_encoder_val = vectorizer_text.transform(self.df_val[text_col].fillna('').astype(str)).toarray()
                
                self.text_encoder_list_train.append(text_encoder_train)
                self.text_encoder_list_val.append(text_encoder_val)
                self.vectorizer_text.append(vectorizer_text)
            
            # Векторизация числовых данных
            self.vectorizer_scalar = []
            self.data_encoder_list_train = []
            self.data_encoder_list_val = []
            
            for data_col in self.data_column:
                vectorizer_data = StandardScaler()
                
                data_encoder_train = vectorizer_data.fit_transform(self.df_train[data_col].fillna(0).values.reshape(-1, 1))
                data_encoder_val = vectorizer_data.transform(self.df_val[data_col].fillna(0).values.reshape(-1, 1))
                
                self.data_encoder_list_train.append(data_encoder_train)
                self.data_encoder_list_val.append(data_encoder_val)
                self.vectorizer_scalar.append(vectorizer_data)
            
            # Объединение текстовых и числовых данных
            text_features_train = np.hstack(self.text_encoder_list_train)
            text_features_val = np.hstack(self.text_encoder_list_val)
            data_features_train = np.hstack(self.data_encoder_list_train)
            data_features_val = np.hstack(self.data_encoder_list_val)
            
            self.X_train = np.hstack([text_features_train, data_features_train])
            self.X_val = np.hstack([text_features_val, data_features_val])
        else:
            raise ValueError("Должен быть указан хотя бы один из параметров: text_column или data_column")
        
        # Кодирование меток
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.df_train[self.label_column])
        self.y_val = self.label_encoder.transform(self.df_val[self.label_column])
        
        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        
        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes
    
    # Векторизация без отдельного набора данных
    def vectorize_no_val_path(self): 

        if self.text_column is None and self.data_column is not None:
            # Векторизация числовых признаков
            X_all, self.vectorizer_scalar = vectorize_data(data_column=self.data_column, 
                                                                       df_train=self.df_train)
            
        elif self.data_column is None and self.text_column is not None:
            X_all, self.vectorizer_text = vectorize_text(text_column=self.text_column,
                                                                     df_train=self.df_train)
            
        elif self.text_column is not None and self.data_column is not None:               
            # Векторизация текста            
            self.text_encoder, self.vectorizer_text = vectorize_text(text_column=self.text_column, 
                                                                     df_train=self.df_train)
            # Векторизация числовых признаков
            self.scalar_encoder, self.vectorizer_scalar = vectorize_data(data_column=self.data_column,
                                                                         df_train=self.df_train)
            # Объединение тикера и числовых признаков
            X_all = np.hstack([self.text_encoder, self.scalar_encoder])
        
        else:
            raise ValueError("Должен быть указан хотя бы один из параметров: text_column или data_column")

        # Кодирование меток
        self.label_encoder = LabelEncoder()
        y_all = self.label_encoder.fit_transform(self.df_train[self.label_column])        

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
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

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
        trainer = Trainer(model=self.model,
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
        metrics = trainer.train(num_epochs=num_epochs,
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
                                crit_name=crit_name
                                )

        # Визуализация метрик
        self.visualize_metrics(metrics)

    def visualize_metrics(self, metrics):
        # Визуализация метрик обучения
        self.mtx.confus_matrix(last_labels=metrics['all_labels'][-1],
                               last_preds=metrics['all_preds'][-1],
                               label_encoder=self.label_encoder.classes_,
                               save_folder_path=self.new_folder_path,
                               plt_name="confusion_matrix_last_model"
                               )

        self.mtx.train_val_loss(epoch=metrics['epoch'],
                                train_loss=metrics['train_loss'],
                                val_loss=metrics['val_loss'],
                                save_folder_path=self.new_folder_path
                                )

        self.mtx.train_val_acc(epoch=metrics['epoch'],
                               train_acc=metrics['train_acc'],
                               val_acc=metrics['val_acc'],
                               save_folder_path=self.new_folder_path
                               )

        self.mtx.f1score(epoch=metrics['epoch'],
                         f1_score=metrics['f1_score'],
                         save_folder_path=self.new_folder_path
                         )

        self.mtx.dataset_counts(data_path=self.train_path,
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
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            self.pred = torch.argmax(self.model(X), dim=1).item()
            self.class_name = self.checkpoint['label_encoder'].inverse_transform([self.pred])[0]

        return self.class_name


class Regression:
    def __init__(self, data_column=None, label_column=None, split_ratio=0.2, random_state=42):
        self.data_column = data_column
        self.label_column = label_column
        self.split_ratio = split_ratio
        self.random_state = random_state
        
        self.df_train = None
        self.df_val = None
        self.vectorizer_scalar = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.input_size = None
        self.output_size = None
        
        self.model = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.metrics = None
        self.new_folder_path = None

    # Выбор функции потерь для регрессии
    def crit(self, crit_name):
        if crit_name == 'MSELoss':
            criterion = nn.MSELoss()
            return criterion
        elif crit_name == 'L1Loss':
            criterion = nn.L1Loss()
            return criterion
        elif crit_name == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss()
            return criterion
        else:
            print("Неизвестная функция потерь для регрессии")
            return nn.MSELoss()
        
    # Выбор оптимизатора
    def optimaze(self, optim_name, params, lr):
        if optim_name == 'Adam':
            optimizer = optim.Adam(params=params, lr=lr)
            return optimizer
        elif optim_name == 'RAdam':
            optimizer = optim.RAdam(params=params, lr=lr)
            return optimizer
        elif optim_name == 'SGD':
            optimizer = optim.SGD(params=params, lr=lr)
            return optimizer
        elif optim_name == 'RMSprop':
            optimizer = optim.RMSprop(params=params, lr=lr)
            return optimizer
        elif optim_name == 'Adadelta':
            optimizer = optim.Adadelta(params=params, lr=lr)
            return optimizer
        
    # Векторизация без отдельного набора данных
    def vectorize_no_val_path(self):
        self.vectorizer_scalar = StandardScaler()
        X_all = self.vectorizer_scalar.fit_transform(self.df_train[self.data_column].values)            

        # Значения напрямую (ТОЛЬКО ДЛЯ РЕГРЕССИИ!!)
        y_all = self.df_train[self.label_column].values

        # Разделение данных на обучающую и валидационную выборку
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_all, 
                                                                              y_all, 
                                                                              test_size=self.split_ratio, 
                                                                              random_state=self.random_state)

        self.input_size = self.X_train.shape[1]
        self.output_size = 1

        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.output_size

    # Векторизация с отдельным валидационным набором данных
    def vectorize_with_val_path(self):
        self.df_val = pd.read_csv(self.val_path)
        
        # Масштабирование признаков
        self.vectorizer_scalar = StandardScaler()
        self.X_train = self.vectorizer_scalar.fit_transform(
            self.df_train[self.data_column].values)
        self.X_val = self.vectorizer_scalar.transform(
            self.df_val[self.data_column].values)

        # Целевые переменные
        self.y_train = self.df_train[self.label_column].values
        self.y_val = self.df_val[self.label_column].values

        self.input_size = self.X_train.shape[1]
        self.output_size = 1

        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.output_size

    def train(self, model_name, train_path, val_path, num_epochs, hidden_size=256, num_layers=1, batch_size=128, learning_rate=0.001, dropout=0, optim_name='Adam', crit_name='MSELoss'):
        """Обучение модели регрессии с заданными параметрами"""
        
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
        headers = ["epoch",
                   "train_loss",
                   "val_loss",
                   "train_mae",
                   "val_mae",
                   "train_rmse",
                   "val_rmse"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)

        # Создание DataLoader
        train_dataset = RegressionDataset(self.X_train, self.y_train)
        val_dataset = RegressionDataset(self.X_val, self.y_val)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

        # Инициализация модели
        self.model = self.model_name(self.input_size,
                                     hidden_size,
                                     num_layers,
                                     self.output_size,
                                     dropout)
        # Инициализация функции потерь
        criterion = self.crit(crit_name=self.crit_name)
        # Инициализация оптимизатора
        optimizer = self.optimaze(optim_name=self.optim_name,
                                  params=self.model.parameters(),
                                  lr=learning_rate)
        # Инициализация GPU (CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(device)

        # Создание и запуск тренера регрессии
        trainer = RegressionTrainer(model=self.model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    device=device,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    vectorizer_scalar=self.vectorizer_scalar,
                                    new_folder_path=self.new_folder_path
                                    )

        # Один вызов train() и сохранение метрик
        metrics = trainer.train(num_epochs=num_epochs,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                input_size=self.input_size,
                                output_size=self.output_size,
                                train_path=train_path,
                                label_column=self.label_column,
                                dropout=dropout,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                optim_name=optim_name,
                                crit_name=crit_name
                                )

        # Запуск визуализация метрик регрессии
        self.visualize_metrics(metrics)
    # Визуализация метрик регрессии
    def visualize_metrics(self, metrics):
        """Визуализация метрик регрессии"""
        
        self.mtx.train_val_loss(epoch=metrics['epoch'],
                                train_loss=metrics['train_loss'],
                                val_loss=metrics['val_loss'],
                                save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_mae(epoch=metrics['epoch'],
                               train_mae=metrics['train_mae'],
                               val_mae=metrics['val_mae'],
                               save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_rmse(epoch=metrics['epoch'],
                                train_rmse=metrics['train_rmse'],
                                val_rmse=metrics['val_rmse'],
                                save_folder_path=self.new_folder_path)
        
        # График распределения предсказаний vs реальных значений
        self.mtx.regression_scatter(metrics=metrics,
                                   save_folder_path=self.new_folder_path)

    def predict(self, model_path, numeric):
        """Инференс модели регрессии"""
        self.model_path = f"{model_path}"
        self.numeric = numeric
        
        # Загружаем на CPU
        self.checkpoint = torch.load(f=self.model_path,
                                     map_location='cpu',
                                     weights_only=False
                                     )  
        self.model = self.checkpoint['model'] 
        self.model.eval()  

        X = self.checkpoint['vectorizer_scalar'].transform([self.numeric])
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            self.pred = self.model(X).item()

        return self.pred