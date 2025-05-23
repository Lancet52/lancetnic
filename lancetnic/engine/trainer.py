import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lancetnic.utils import Metrics, dir


# Датасет для бинарной классификиции
class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Binary:
    def __init__(self, text_column='description', label_column='category', split_ratio=0.2, random_state=42):
        self.text_column = text_column
        self.label_column = label_column

        self.df_train = None
        self.df_val = None
        self.vectorizer = None
        self.X_train = None
        self.X_val = None
        self.label_encoder = None
        self.y_train = None
        self.y_val = None
        self.input_size = None
        self.num_epochs = None
        self.num_classes = None

        self.model = None
        self.criterion = None
        self.optimizer = None
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

    def vectorize_with_val_path(self):
        self.df_val = pd.read_csv(self.val_path)
        # Векторизация текста
        self.vectorizer = TfidfVectorizer()
        self.X_train = self.vectorizer.fit_transform(
            self.df_train[self.text_column]).toarray()
        self.X_val = self.vectorizer.transform(
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
    
    def vectorize_no_val_path(self):
        # Векторизация текста
        self.vectorizer = TfidfVectorizer()
        X_all = self.vectorizer.fit_transform(
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

    def train(self, model_name, train_path, val_path, num_epochs, hidden_size=256, num_layers=1, batch_size=128, learning_rate=0.001, dropout=0):
        """Обучение модели"""
        # Загрузка и предобработка данных
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epochs = num_epochs
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
        # Создание папки для сохранения результатов обучения
        self.new_folder_path = dir()

        # Создание файла для результатов
        headers = ["epoch", "train_loss", "train_acc, %",
                   "val_loss", "val_acc, %", "F1_score"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)

        # Создание DataLoader
        train_dataset = BinaryDataset(self.X_train, self.y_train)
        val_dataset = BinaryDataset(self.X_val, self.y_val)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Инициализация модели
        self.model = self.model_name(
            self.input_size, hidden_size, num_layers, self.num_classes, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Подготовка к обучению
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.best_val_loss = float('inf')
        self.metrics = {'epoch': [], 'train_loss': [], 'val_loss': [],
                        'train_acc': [], 'val_acc': [], 'f1_score': [],
                        'all_preds': [], 'all_labels': []
                        }

        for epoch in range(self.num_epochs):
            # Обучение
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.float()  # Добавление преобразование типа

                outputs = self.model(inputs.unsqueeze(1))
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Валидация
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    inputs = inputs.float()  # Добавление преобразование типа

                    outputs = self.model(inputs.unsqueeze(1))
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Вычисление метрик
            train_loss_epoch = train_loss / len(self.train_loader)
            val_loss_epoch = val_loss / len(self.val_loader)
            train_acc_epoch = train_correct / train_total
            val_acc_epoch = val_correct / val_total
            f1 = f1_score(all_labels, all_preds, average='weighted')

            # Сохранение метрик
            self.metrics['epoch'].append(epoch + 1)
            self.metrics['train_loss'].append(train_loss_epoch)
            self.metrics['val_loss'].append(val_loss_epoch)
            self.metrics['train_acc'].append(train_acc_epoch)
            self.metrics['val_acc'].append(val_acc_epoch)
            self.metrics['f1_score'].append(f1)
            self.metrics['all_preds'].append(all_preds)
            self.metrics['all_labels'].append(all_labels)

            # Сохранение лучшей модели
            if val_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_loss_epoch
                # Сохранение матрицы ошибок для лучшей модели
                self.mtx.confus_matrix(last_labels=all_labels,
                                       last_preds=all_preds,
                                       label_encoder=self.label_encoder.classes_,
                                       save_folder_path=self.new_folder_path,
                                       plt_name="confusion_matrix_best_model")
                torch.save({
                    'model': self.model,
                    'input_size': self.input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'num_classes': self.num_classes,
                    'vectorizer': self.vectorizer,
                    'label_encoder': self.label_encoder,
                    'epoch': epoch,
                    'val_loss': val_loss_epoch,
                    'val_acc': val_acc_epoch
                }, f"{self.new_folder_path}/best_model.pt")

            # Сохранение последней модели (после каждой эпохи)
            torch.save({
                'model': self.model,
                'input_size': self.input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_classes': self.num_classes,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'epoch': epoch,
                'val_loss': val_loss_epoch,
                'val_acc': val_acc_epoch
            }, f"{self.new_folder_path}/last_model.pt")

            # Запись результатов обучения в CSV
            csv_data = {
                "epoch": epoch + 1,
                "train_loss": f"{train_loss_epoch:.4f}",
                "train_acc, %": f"{100 * train_acc_epoch:.2f}",
                "val_loss": f"{val_loss_epoch:.4f}",
                "val_acc, %": f"{100 * val_acc_epoch:.2f}",
                "F1_score": f"{100 * f1:.2f}"
            }
            pd.DataFrame([csv_data]).to_csv(
                self.csv_path, mode='a', header=False, index=False)

            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(
                f"Train Loss: {train_loss_epoch:.4f} | Train Acc: {100 * train_acc_epoch:.2f}%")
            print(
                f"Val Loss: {val_loss_epoch:.4f} | Val Acc: {100 * val_acc_epoch:.2f}% | F1-score: {100 * f1:.2f}%")
            print("-" * 50)

        print("Обучение завершено!")
        print(
            f"Лучшая модель сохранена в '{self.new_folder_path}\\best_model.pt' с val loss: {self.best_val_loss:.4f}")
        print(
            f"Последняя модель сохранена в '{self.new_folder_path}\\last_model.pt'")

        # Визуализация метрик

        self.mtx.confus_matrix(last_labels=self.metrics['all_labels'][-1],
                               last_preds=self.metrics['all_preds'][-1],
                               label_encoder=self.label_encoder.classes_,
                               save_folder_path=self.new_folder_path,
                               plt_name="confusion_matrix_last_model")

        self.mtx.train_val_loss(epoch=self.metrics['epoch'],
                                train_loss=self.metrics['train_loss'],
                                val_loss=self.metrics['val_loss'],
                                save_folder_path=self.new_folder_path)

        self.mtx.train_val_acc(epoch=self.metrics['epoch'],
                               train_acc=self.metrics['train_acc'],
                               val_acc=self.metrics['val_acc'],
                               save_folder_path=self.new_folder_path)

        self.mtx.f1score(epoch=self.metrics['epoch'],
                         f1_score=self.metrics['f1_score'],
                         save_folder_path=self.new_folder_path)
        
        self.mtx.dataset_counts(data_path=self.train_path,
                                label_column=self.label_column,
                                save_folder_path=self.new_folder_path)
