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


#Датасет для бинарной классификиции
class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class Binary:
    def __init__(self, text_column='description', label_column='category'):
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

    def train(self, model_name, train_path, val_path, num_epochs, hidden_size=256, num_layers=1, batch_size=128, learning_rate=0.001):
        """Обучение модели"""
        # Загрузка и предобработка данных
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epochs = num_epochs
        self.df_train = pd.read_csv(self.train_path)
        self.df_val = pd.read_csv(self.val_path)
        
        # Векторизация текста
        self.vectorizer = TfidfVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.df_train[self.text_column]).toarray()
        self.X_val = self.vectorizer.transform(self.df_val[self.text_column]).toarray()
        
        # Кодирование меток
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.df_train[self.label_column])
        self.y_val = self.label_encoder.transform(self.df_val[self.label_column])
        
        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        
        # Настройка обучения
        # Создание папки для сохранения результатов обучения
        base_dir = os.getcwd()
        base_dir_runs = os.path.join(base_dir, "runs")
        if not os.path.exists(base_dir_runs):
            os.makedirs(base_dir_runs)
        
        train_folders = []
        for folder in os.listdir(base_dir_runs):
            if os.path.isdir(os.path.join(base_dir_runs, folder)) and folder.startswith("train_"):
                train_folders.append(folder)
        
        folder_numbers = []
        for f in train_folders:
            try:
                num = int(f.split("_")[-1])
                folder_numbers.append(num)
            except ValueError:
                continue
        
        next_number = max(folder_numbers) + 1 if folder_numbers else 0
        new_folder = f"train_{next_number}"
        self.new_folder_path = os.path.join(base_dir_runs, new_folder)
        
        os.makedirs(self.new_folder_path, exist_ok=True)
        print(f"Создана папка: {self.new_folder_path}")
        
        # Создание файла для результатов
        headers = ["epoch", "train_loss", "train_acc, %", 
                  "val_loss", "val_acc, %", "F1_score"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        # Создание DataLoader
        train_dataset = BinaryDataset(self.X_train, self.y_train)
        val_dataset = BinaryDataset(self.X_val, self.y_val)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Инициализация модели
        self.model = self.model_name(self.input_size, hidden_size, num_layers, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Подготовка к обучению
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.best_val_loss = float('inf')
        self.metrics = {'epoch': [], 'train_loss': [], 'val_loss': [],
                        'train_acc': [], 'val_acc': [], 'f1_score': [],
                        'all_preds': [], 'all_labels': []
                        }
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.float()  # Добавляем преобразование типа

                outputs = self.model(inputs.unsqueeze(1))
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = inputs.float()  # Добавляем преобразование типа

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

            # Сохраняем лучшую модель
            if val_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_loss_epoch
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
                }, f"{self.new_folder_path}/best_model.pth")

            # Сохраняем последнюю модель после каждой эпохи
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
            }, f"{self.new_folder_path}/last_model.pth")

            # Запись в CSV
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
            print(f"Train Loss: {train_loss_epoch:.4f} | Train Acc: {100 * train_acc_epoch:.2f}%")
            print(f"Val Loss: {val_loss_epoch:.4f} | Val Acc: {100 * val_acc_epoch:.2f}% | F1-score: {100 * f1:.2f}%")
            print("-" * 50)

        print("Обучение завершено!")
        print(f"Лучшая модель сохранена в '{self.new_folder_path}/best_model.pth' с val loss: {self.best_val_loss:.4f}")
        print(f"Последняя модель сохранена в '{self.new_folder_path}/last_model.pth'")

        # Визуализация метрик
        self._plot_metrics()

    def _plot_metrics(self):
        """Визуализация метрик обучения"""
        # Визуализация F1-score
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['epoch'], self.metrics['f1_score'],
                label='Validation F1-score', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.title('F1-score during Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.new_folder_path}/f1_score_plot.png")
        plt.close()

        # Дополнительная визуализация других метрик
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['epoch'], self.metrics['train_acc'], label='Train Accuracy')
        plt.plot(self.metrics['epoch'], self.metrics['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.metrics['epoch'], self.metrics['f1_score'], label='Validation F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.title('Validation F1-score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.new_folder_path}/training_metrics.png")
        plt.close()

        # Confusion matrix для последней эпохи
        last_labels = self.metrics['all_labels'][-1]  # Берем последние сохраненные
        last_preds = self.metrics['all_preds'][-1]
        conf_matrix = confusion_matrix(last_labels, last_preds)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(f"{self.new_folder_path}/confusion_matrix.png")
        plt.close()