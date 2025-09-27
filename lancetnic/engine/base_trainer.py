import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import f1_score
from lancetnic.utils import Metrics
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device, train_loader, val_loader, label_encoder, vectorizer_text, vectorizer_scalar, new_folder_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_encoder = label_encoder
        self.vectorizer_text = vectorizer_text
        self.vectorizer_scalar = vectorizer_scalar
        self.new_folder_path = new_folder_path
        self.best_val_loss = float('inf')
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'f1_score': [],
            'all_preds': [],
            'all_labels': []
        }
        self.mtx = Metrics()

    def save_hyperparameters(self, hyperparams):
        path_config = f"{self.new_folder_path}/hyperparams.yaml"
        with open(path_config, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
        print(f"Гиперпараметры сохранены в: {path_config}/hyperparams.yaml")

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.float()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        return train_loss, train_correct, train_total

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return val_loss, val_correct, val_total, all_preds, all_labels

    def calculate_metrics(self, train_loss, train_correct, train_total, 
                         val_loss, val_correct, val_total, all_preds, all_labels):
        train_loss_epoch = train_loss / len(self.train_loader)
        val_loss_epoch = val_loss / len(self.val_loader)
        train_acc_epoch = train_correct / train_total
        val_acc_epoch = val_correct / val_total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1

    def save_metrics(self, epoch, train_loss_epoch, val_loss_epoch, 
                    train_acc_epoch, val_acc_epoch, f1, all_preds, all_labels):
        self.metrics['epoch'].append(epoch + 1)
        self.metrics['train_loss'].append(train_loss_epoch)
        self.metrics['val_loss'].append(val_loss_epoch)
        self.metrics['train_acc'].append(train_acc_epoch)
        self.metrics['val_acc'].append(val_acc_epoch)
        self.metrics['f1_score'].append(f1)
        self.metrics['all_preds'].append(all_preds)
        self.metrics['all_labels'].append(all_labels)

    def save_model(self, epoch, val_loss_epoch, val_acc_epoch, hidden_size, num_layers, input_size, num_classes):
        if val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self.mtx.confus_matrix(
                last_labels=self.metrics['all_labels'][-1],
                last_preds=self.metrics['all_preds'][-1],
                label_encoder=self.label_encoder.classes_,
                save_folder_path=self.new_folder_path,
                plt_name="confusion_matrix_best_model"
            )
            torch.save({
                'model': self.model,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_classes': num_classes,
                'vectorizer_text': self.vectorizer_text,
                'vectorizer_scalar': self.vectorizer_scalar,
                'label_encoder': self.label_encoder,
                'epoch': epoch,
                'val_loss': val_loss_epoch,
                'val_acc': val_acc_epoch
            }, f"{self.new_folder_path}/best_model.pt")

        torch.save({
            'model': self.model,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'vectorizer_text': self.vectorizer_text,
            'vectorizer_scalar': self.vectorizer_scalar,
            'label_encoder': self.label_encoder,
            'epoch': epoch,
            'val_loss': val_loss_epoch,
            'val_acc': val_acc_epoch
        }, f"{self.new_folder_path}/last_model.pt")

    def log_results(self, epoch, num_epochs, train_loss_epoch, train_acc_epoch, 
                   val_loss_epoch, val_acc_epoch, f1):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss_epoch:.4f} | Train Acc: {100 * train_acc_epoch:.2f}%")
        print(f"Val Loss: {val_loss_epoch:.4f} | Val Acc: {100 * val_acc_epoch:.2f}% | F1-score: {100 * f1:.2f}%")
        print("-" * 50)

    def save_to_csv(self, epoch, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch, f1):
        csv_path = f"{self.new_folder_path}/result.csv"
        csv_data = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss_epoch:.4f}",
            "train_acc, %": f"{100 * train_acc_epoch:.2f}",
            "val_loss": f"{val_loss_epoch:.4f}",
            "val_acc, %": f"{100 * val_acc_epoch:.2f}",
            "F1_score": f"{100 * f1:.2f}"
        }
        pd.DataFrame([csv_data]).to_csv(csv_path, mode='a', header=False, index=False)

    

    def train(self, num_epochs, hidden_size, num_layers, input_size, num_classes, train_path, label_column, dropout, batch_size, learning_rate, optim_name, crit_name):
        hyperparams = {
                "model_params": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": num_classes,
                    "dropout": dropout
                },

                "train_params": {
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "optimizer": optim_name,
                    "criterion": crit_name
                }
            }
        self.save_hyperparameters(hyperparams=hyperparams)
        for epoch in range(num_epochs):
            
            # Фаза обучения
            train_loss, train_correct, train_total = self.train_epoch(epoch, num_epochs)
            
            # Валидация
            val_loss, val_correct, val_total, all_preds, all_labels = self.validate_epoch()
            
            # Вычисление метрик
            train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1 = self.calculate_metrics(
                train_loss, train_correct, train_total, 
                val_loss, val_correct, val_total, 
                all_preds, all_labels
            )
            
            # Сохранение метрик
            self.save_metrics(
                epoch, train_loss_epoch, val_loss_epoch, 
                train_acc_epoch, val_acc_epoch, f1, 
                all_preds, all_labels
            )
            
            # Сохранение модели
            self.save_model(
                epoch, val_loss_epoch, val_acc_epoch, 
                hidden_size, num_layers, input_size, num_classes
            )
            
            # Результаты
            self.log_results(
                epoch, num_epochs, train_loss_epoch, train_acc_epoch, 
                val_loss_epoch, val_acc_epoch, f1
            )
            
            # Сохранение в CSV
            self.save_to_csv(
                epoch, train_loss_epoch, train_acc_epoch, 
                val_loss_epoch, val_acc_epoch, f1
            )

        print("Обучение завершено!")
        print(f"Лучшая модель сохранена в '{self.new_folder_path}\\best_model.pt' с val loss: {self.best_val_loss:.4f}")
        print(f"Последняя модель сохранена в '{self.new_folder_path}\last_model.pt'")
        
        
        return self.metrics