import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from lancetnic.utils import Metrics
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ClassificationTrainer:
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
        self.metrics = {'epoch': [],
                        'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': [],
                        'val_precision': [],
                        'val_recall': [],
                        'f1_score': [],
                        'all_preds': [],
                        'all_labels': []
                        }
        self.mtx = Metrics()

    def save_hyperparameters(self, hyperparams):
        path_config = f"{self.new_folder_path}\\hyperparams.yaml"
        with open(path_config, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
        print(f"Гиперпараметры сохранены в: {path_config}")

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

                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return val_loss, val_correct, val_total, all_preds, all_labels

    def calculate_metrics(self, train_loss, train_correct, train_total, val_loss, val_correct, val_total, all_preds, all_labels):
        train_loss_epoch = train_loss / len(self.train_loader)
        val_loss_epoch = val_loss / len(self.val_loader)
        train_acc_epoch = train_correct / train_total
        val_acc_epoch = val_correct / val_total
        true_labels = np.concatenate(all_labels).ravel()
        pred_labels = np.concatenate(all_preds).ravel()
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        val_precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        val_recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)


        return train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall

    def save_metrics(self, epoch, train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall, all_preds, all_labels):
        self.metrics['epoch'].append(epoch + 1)
        self.metrics['train_loss'].append(train_loss_epoch)
        self.metrics['val_loss'].append(val_loss_epoch)
        self.metrics['train_acc'].append(train_acc_epoch)
        self.metrics['val_acc'].append(val_acc_epoch)
        self.metrics['f1_score'].append(f1)
        self.metrics['val_precision'].append(val_precision)
        self.metrics['val_recall'].append(val_recall)
        self.metrics['all_preds'].append(np.concatenate(all_preds).ravel())
        self.metrics['all_labels'].append(np.concatenate(all_labels).ravel())

    def save_model(self, epoch, val_loss_epoch, val_acc_epoch, hidden_size, num_layers, input_size, num_classes):
        if val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self.mtx.confus_matrix(last_labels=self.metrics['all_labels'][-1],
                                   last_preds=self.metrics['all_preds'][-1],
                                   label_encoder=self.label_encoder.classes_,
                                   save_folder_path=self.new_folder_path,
                                   plt_name="confusion_matrix_best_model"
                                   )
            torch.save({'model': self.model,
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

        torch.save({'model': self.model,
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

    def log_results(self, epoch, num_epochs, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch, f1, val_precision, val_recall):
        print(f"Epoch [{epoch + 1}/{num_epochs}] ")
        print(f"Train Loss: {train_loss_epoch:.4f} | Train Acc: {100 * train_acc_epoch:.2f}% ")
        print(f"Val Loss: {val_loss_epoch:.4f} | Val Acc: {100 * val_acc_epoch:.2f}% | Val Prec: {100 * val_precision:.2f}% | Val Rec: {100 * val_recall:.2f}% | F1: {100 * f1:.2f}% ")
        print("-" * 50)

    def save_to_csv(self, epoch, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch, f1, val_precision, val_recall):
        csv_path = f"{self.new_folder_path}/result.csv"
        csv_data = {"epoch": epoch + 1,
                    "train_loss": f"{train_loss_epoch:.4f}",
                    "train_acc, %": f"{100 * train_acc_epoch:.2f}",
                    "val_loss": f"{val_loss_epoch:.4f}",
                    "val_acc, %": f"{100 * val_acc_epoch:.2f}",
                    "val_precision, %": f"{100 * val_precision:.2f}",
                    "val_recall, %": f"{100 * val_recall:.2f}",
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
            train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall= self.calculate_metrics(train_loss, train_correct, train_total, val_loss, val_correct, val_total, all_preds, all_labels)
            
            # Сохранение метрик
            self.save_metrics(epoch=epoch,
                              train_loss_epoch=train_loss_epoch,
                              val_loss_epoch=val_loss_epoch,
                              train_acc_epoch=train_acc_epoch,
                              val_acc_epoch=val_acc_epoch,
                              f1=f1,
                              val_precision=val_precision,
                              val_recall=val_recall,
                              all_preds=all_preds,
                              all_labels=all_labels)
            
            # Сохранение модели
            self.save_model(epoch=epoch,
                            val_loss_epoch=val_loss_epoch,
                            val_acc_epoch=val_acc_epoch,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            input_size=input_size,
                            num_classes=num_classes)
            
            # Результаты
            self.log_results(epoch=epoch,
                             num_epochs=num_epochs,
                             train_loss_epoch=train_loss_epoch,
                             train_acc_epoch=train_acc_epoch,
                             val_loss_epoch=val_loss_epoch,
                             val_acc_epoch=val_acc_epoch,
                             f1=f1,
                             val_precision=val_precision,
                             val_recall=val_recall
                             )
            
            # Сохранение в CSV
            self.save_to_csv(epoch=epoch,
                             train_loss_epoch=train_loss_epoch,
                             train_acc_epoch=train_acc_epoch,
                             val_loss_epoch=val_loss_epoch,
                             val_acc_epoch=val_acc_epoch,
                             f1=f1,
                             val_precision=val_precision,
                             val_recall=val_recall
                             )

        print("Обучение завершено!")
        print(f"Лучшая модель сохранена в '{self.new_folder_path}\\best_model.pt' с val loss: {self.best_val_loss:.4f}")
        print(f"Последняя модель сохранена в '{self.new_folder_path}\\last_model.pt'")
        
        
        return self.metrics
    
class RegressionTrainer:
    def __init__(self, model, criterion, optimizer, device, train_loader, val_loader, vectorizer_text, vectorizer_scalar, new_folder_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vectorizer_text = vectorizer_text
        self.vectorizer_scalar = vectorizer_scalar
        self.new_folder_path = new_folder_path
        self.best_val_loss = float('inf')
        self.metrics = {'epoch': [],
                        'train_loss': [],
                        'val_loss': [],
                        'train_mae': [],
                        'val_mae': [],
                        'train_rmse': [],
                        'val_rmse': [],
                        'all_val_preds': [],
                        'all_val_labels': []
                        }

    def save_hyperparameters(self, hyperparams):
        path_config = f"{self.new_folder_path}\\hyperparams.yaml"
        with open(path_config, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
        print(f"Гиперпараметры сохранены в: {path_config}")

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            all_train_preds.extend(outputs.cpu().detach().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        return train_loss, all_train_preds, all_train_labels

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                all_val_preds.extend(outputs.cpu().numpy().flatten())
                all_val_labels.extend(labels.cpu().numpy().flatten())

        return val_loss, all_val_preds, all_val_labels

    def calculate_metrics(self, train_loss, val_loss, train_preds, train_labels, val_preds, val_labels):
        train_loss_epoch = train_loss / len(self.train_loader)
        val_loss_epoch = val_loss / len(self.val_loader)
        
        train_mae = mean_absolute_error(train_labels, train_preds)
        val_mae = mean_absolute_error(val_labels, val_preds)
        
        train_rmse = np.sqrt(mean_squared_error(train_labels, train_preds))
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))

        return train_loss_epoch, val_loss_epoch, train_mae, val_mae, train_rmse, val_rmse

    def save_metrics(self, epoch, train_loss_epoch, val_loss_epoch, train_mae, val_mae, train_rmse, val_rmse, val_preds, val_labels):
        self.metrics['epoch'].append(epoch + 1)
        self.metrics['train_loss'].append(train_loss_epoch)
        self.metrics['val_loss'].append(val_loss_epoch)
        self.metrics['train_mae'].append(train_mae)
        self.metrics['val_mae'].append(val_mae)
        self.metrics['train_rmse'].append(train_rmse)
        self.metrics['val_rmse'].append(val_rmse)
        self.metrics['all_val_preds'].append(val_preds)
        self.metrics['all_val_labels'].append(val_labels)

    def save_model(self, epoch, val_loss_epoch, hidden_size, num_layers, input_size, output_size):
        if val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            torch.save({'model': self.model,
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'output_size': output_size,
                        'vectorizer_text': self.vectorizer_text,
                        'vectorizer_scalar': self.vectorizer_scalar,
                        'epoch': epoch,
                        'val_loss': val_loss_epoch
                        }, f"{self.new_folder_path}/best_model.pt")

        torch.save({'model': self.model,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'vectorizer_text': self.vectorizer_text,
                    'vectorizer_scalar': self.vectorizer_scalar,
                    'epoch': epoch,
                    'val_loss': val_loss_epoch
                    }, f"{self.new_folder_path}/last_model.pt")

    def log_results(self, epoch, num_epochs, train_loss_epoch, val_loss_epoch, train_mae, val_mae):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss_epoch:.4f} | Train MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss_epoch:.4f} | Val MAE: {val_mae:.4f}")
        print("-" * 50)

    def save_to_csv(self, epoch, train_loss_epoch, val_loss_epoch, train_mae, val_mae, train_rmse, val_rmse):
        csv_path = f"{self.new_folder_path}/result.csv"
        csv_data = {"epoch": epoch + 1,
                    "train_loss": f"{train_loss_epoch:.4f}",
                    "val_loss": f"{val_loss_epoch:.4f}",
                    "train_mae": f"{train_mae:.4f}",
                    "val_mae": f"{val_mae:.4f}",
                    "train_rmse": f"{train_rmse:.4f}",
                    "val_rmse": f"{val_rmse:.4f}"
                    }
        pd.DataFrame([csv_data]).to_csv(csv_path, mode='a', header=False, index=False)

    def train(self, num_epochs, hidden_size, num_layers, input_size, output_size, train_path, label_column, dropout, batch_size, learning_rate, optim_name, crit_name):
        hyperparams = {
            "model_params": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "output_size": output_size,
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
            train_loss, train_preds, train_labels = self.train_epoch(epoch=epoch,
                                                                     num_epochs=num_epochs)
            
            # Валидация
            val_loss, val_preds, val_labels = self.validate_epoch()
            
            # Вычисление метрик
            train_loss_epoch, val_loss_epoch, train_mae, val_mae, train_rmse, val_rmse = self.calculate_metrics(train_loss=train_loss,
                                                                                                                val_loss=val_loss,
                                                                                                                train_preds=train_preds,
                                                                                                                train_labels=train_labels,
                                                                                                                val_preds=val_preds,
                                                                                                                val_labels=val_labels)
            
            # Сохранение метрик
            self.save_metrics(epoch=epoch,
                              train_loss_epoch=train_loss_epoch,
                              val_loss_epoch=val_loss_epoch,
                              train_mae=train_mae,
                              val_mae=val_mae,
                              train_rmse=train_rmse,
                              val_rmse=val_rmse,
                              val_preds=val_preds,
                              val_labels=val_labels
                              )
            
            # Сохранение модели
            self.save_model(epoch=epoch,
                            val_loss_epoch=val_loss_epoch,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            input_size=input_size,
                            output_size=output_size
                            )
            
            # Результаты
            self.log_results(epoch=epoch,
                             num_epochs=num_epochs,
                             train_loss_epoch=train_loss_epoch,
                             val_loss_epoch=val_loss_epoch,
                             train_mae=train_mae,
                             val_mae=val_mae
                             )
            
            # Сохранение в CSV
            self.save_to_csv(epoch=epoch,
                             train_loss_epoch=train_loss_epoch,
                             val_loss_epoch=val_loss_epoch,
                             train_mae=train_mae,
                             val_mae=val_mae,
                             train_rmse=train_rmse,
                             val_rmse=val_rmse
                             )

        print("Обучение завершено!")
        print(f"Лучшая модель сохранена в '{self.new_folder_path}/best_model.pt' с val loss: {self.best_val_loss:.4f}")
        print(f"Последняя модель сохранена в '{self.new_folder_path}/last_model.pt'")
        
        return self.metrics
class MultiTaskTrainer:
    def __init__(self, model, criterion_class, criterion_reg, optimizer, device,
                 train_loader, val_loader, label_encoder, vectorizer_text, vectorizer_scalar, new_folder_path, loss_ratio_cls, loss_ratio_reg, scaler_reg):
        self.model = model
        self.criterion_class = criterion_class
        self.criterion_reg = criterion_reg
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_encoder = label_encoder
        self.vectorizer_text = vectorizer_text
        self.vectorizer_scalar = vectorizer_scalar
        self.new_folder_path = new_folder_path
        self.best_val_loss = float('inf')
        self.loss_ratio_cls = loss_ratio_cls
        self.loss_ratio_reg = loss_ratio_reg
        self.scaler_reg = scaler_reg
        self.metrics = {'epoch': [],
                        'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': [],
                        'val_precision': [],
                        'val_recall': [],
                        'f1_score': [],
                        'train_mae': [],
                        'val_mae': [],
                        'train_rmse': [],
                        'val_rmse': [],
                        'all_preds': [],
                        'all_labels': [],
                        'all_val_preds': [],
                        'all_val_labels': []}
        
        self.mtx=Metrics()

    def save_hyperparameters(self, hyperparams):
        path_config = f"{self.new_folder_path}\\hyperparams.yaml"
        with open(path_config, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
        print(f"Гиперпараметры сохранены в: {path_config}")

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        all_train_preds_reg, all_train_labels_reg = [], []

        for inputs, labels_class, labels_reg in tqdm(self.train_loader, desc="Training"):
            inputs, labels_class, labels_reg = inputs.to(self.device), labels_class.to(self.device), labels_reg.to(self.device)
            self.optimizer.zero_grad()
            class_out, reg_out = self.model(inputs.float())
            
            loss_c = self.criterion_class(class_out, labels_class)
            loss_r = self.criterion_reg(reg_out, labels_reg)
            loss = self.loss_ratio_cls * loss_c + self.loss_ratio_reg * loss_r
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(class_out.data, 1)
            train_total += labels_class.size(0)
            train_correct += (predicted == labels_class).sum().item()
            
            all_train_preds_reg.extend(reg_out.cpu().detach().numpy().flatten())
            all_train_labels_reg.extend(labels_reg.cpu().numpy().flatten())

        train_mae = mean_absolute_error(all_train_labels_reg, all_train_preds_reg)
        train_rmse = np.sqrt(mean_squared_error(all_train_labels_reg, all_train_preds_reg))
        return train_loss, train_correct, train_total, train_mae, train_rmse

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels_class, labels_reg in tqdm(self.val_loader, desc="Validation"):
                inputs, labels_class, labels_reg = inputs.to(self.device), labels_class.to(self.device), labels_reg.to(self.device)
                class_out, reg_out = self.model(inputs.float())
                
                loss_c = self.criterion_class(class_out, labels_class)
                loss_r = self.criterion_reg(reg_out, labels_reg)
                val_loss += (self.loss_ratio_cls * loss_c + self.loss_ratio_reg * loss_r).item()
                
                _, predicted = torch.max(class_out.data, 1)
                val_total += labels_class.size(0)
                val_correct += (predicted == labels_class).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_class.cpu().numpy())
                all_val_preds.extend(reg_out.cpu().numpy().flatten())
                all_val_labels.extend(labels_reg.cpu().numpy().flatten())

        return val_loss, val_correct, val_total, all_preds, all_labels, all_val_preds, all_val_labels

    def calculate_metrics(self, train_loss, train_correct, train_total, val_loss, val_correct, val_total, all_preds, all_labels, all_val_preds, all_val_labels, train_mae, train_rmse):
        train_loss_epoch = train_loss / len(self.train_loader)
        val_loss_epoch = val_loss / len(self.val_loader)
        train_acc_epoch = train_correct / train_total
        val_acc_epoch = val_correct / val_total
        
        true_labels = np.array(all_labels)
        pred_labels = np.array(all_preds)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        val_precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        val_recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        val_mae = mean_absolute_error(all_val_labels, all_val_preds)
        val_rmse = np.sqrt(mean_squared_error(all_val_labels, all_val_preds))
        
        return train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall, train_mae, train_rmse, val_mae, val_rmse

    def save_metrics(self, epoch, train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall, train_mae, train_rmse, val_mae, val_rmse, all_preds, all_labels, all_val_preds, all_val_labels):
        self.metrics['epoch'].append(epoch + 1)
        self.metrics['train_loss'].append(train_loss_epoch)
        self.metrics['val_loss'].append(val_loss_epoch)
        self.metrics['train_acc'].append(train_acc_epoch)
        self.metrics['val_acc'].append(val_acc_epoch)
        self.metrics['f1_score'].append(f1)
        self.metrics['val_precision'].append(val_precision)
        self.metrics['val_recall'].append(val_recall)
        self.metrics['train_mae'].append(train_mae)
        self.metrics['val_mae'].append(val_mae)
        self.metrics['train_rmse'].append(train_rmse)
        self.metrics['val_rmse'].append(val_rmse)
        self.metrics['all_preds'].append(np.array(all_preds))
        self.metrics['all_labels'].append(np.array(all_labels))
        self.metrics['all_val_preds'].append(all_val_preds)
        self.metrics['all_val_labels'].append(all_val_labels)

    def save_model(self, epoch, val_loss_epoch, hidden_size, num_layers, input_size, num_classes):
        if val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self.mtx.confus_matrix(last_labels=self.metrics['all_labels'][-1],
                                   last_preds=self.metrics['all_preds'][-1],
                                   label_encoder=self.label_encoder.classes_,
                                   save_folder_path=self.new_folder_path,
                                   plt_name="confusion_matrix_best_model"
                                   )
            torch.save({
                'model': self.model, 
                'label_encoder': self.label_encoder,
                'vectorizer_text': self.vectorizer_text, 
                'vectorizer_scalar': self.vectorizer_scalar,
                'scaler_reg': self.scaler_reg,
                'epoch': epoch, 
                'val_loss': val_loss_epoch,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'input_size': input_size,
                'num_classes': num_classes
            }, f"{self.new_folder_path}/best_model.pt")
            
        torch.save({
            'model': self.model, 
            'label_encoder': self.label_encoder,
            'vectorizer_text': self.vectorizer_text, 
            'vectorizer_scalar': self.vectorizer_scalar,
            'scaler_reg': self.scaler_reg,
            'epoch': epoch, 
            'val_loss': val_loss_epoch,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'input_size': input_size,
            'num_classes': num_classes
        }, f"{self.new_folder_path}/last_model.pt")

    def log_results(self, epoch, num_epochs, train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall, train_mae, train_rmse, val_mae, val_rmse):
        print(f"Epoch [{epoch+1}/{num_epochs}] | TrLoss: {train_loss_epoch:.4f} | ValLoss: {val_loss_epoch:.4f} | "
              f"ValAcc: {val_acc_epoch*100:.2f}% | F1: {f1*100:.2f}% | ValMAE: {val_mae:.4f} | ValRMSE: {val_rmse:.4f}")
        print("-" * 50)

    def save_to_csv(self, epoch, train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall, train_mae, train_rmse, val_mae, val_rmse):
        csv_path = f"{self.new_folder_path}/result.csv"
        csv_data = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss_epoch:.4f}",
            "val_loss": f"{val_loss_epoch:.4f}",
            "train_acc, %": f"{100 * train_acc_epoch:.2f}",
            "val_acc, %": f"{100 * val_acc_epoch:.2f}",
            "val_precision, %": f"{100 * val_precision:.2f}",
            "val_recall, %": f"{100 * val_recall:.2f}",
            "F1_score": f"{100 * f1:.2f}",
            "train_mae": f"{train_mae:.4f}",
            "val_mae": f"{val_mae:.4f}",
            "train_rmse": f"{train_rmse:.4f}",
            "val_rmse": f"{val_rmse:.4f}"
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
                "criterion": crit_name if crit_name else "CELoss + MSELoss"
            }
        }
        self.save_hyperparameters(hyperparams=hyperparams)
        
        for epoch in range(num_epochs):
            # Фаза обучения
            train_loss, train_correct, train_total, train_mae, train_rmse = self.train_epoch(epoch, num_epochs)
            
            # Валидация
            val_loss, val_correct, val_total, all_preds, all_labels, all_val_preds, all_val_labels = self.validate_epoch()
            
            # Вычисление метрик
            train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, f1, val_precision, val_recall, train_mae, train_rmse, val_mae, val_rmse = self.calculate_metrics(
                train_loss, train_correct, train_total, val_loss, val_correct, val_total, all_preds, all_labels, all_val_preds, all_val_labels, train_mae, train_rmse
            )
            
            # Сохранение метрик
            self.save_metrics(epoch=epoch,
                              train_loss_epoch=train_loss_epoch,
                              val_loss_epoch=val_loss_epoch,
                              train_acc_epoch=train_acc_epoch,
                              val_acc_epoch=val_acc_epoch,
                              f1=f1,
                              val_precision=val_precision,
                              val_recall=val_recall,
                              train_mae=train_mae,
                              train_rmse=train_rmse,
                              val_mae=val_mae,
                              val_rmse=val_rmse,
                              all_preds=all_preds,
                              all_labels=all_labels,
                              all_val_preds=all_val_preds,
                              all_val_labels=all_val_labels)
            
            # Сохранение модели
            self.save_model(epoch=epoch,
                            val_loss_epoch=val_loss_epoch,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            input_size=input_size,
                            num_classes=num_classes)
            
            # Результаты
            self.log_results(epoch=epoch,
                             num_epochs=num_epochs,
                             train_loss_epoch=train_loss_epoch,
                             val_loss_epoch=val_loss_epoch,
                             train_acc_epoch=train_acc_epoch,
                             val_acc_epoch=val_acc_epoch,
                             f1=f1,
                             val_precision=val_precision,
                             val_recall=val_recall,
                             train_mae=train_mae,
                             train_rmse=train_rmse,
                             val_mae=val_mae,
                             val_rmse=val_rmse)
            
            # Сохранение в CSV
            self.save_to_csv(epoch=epoch,
                             train_loss_epoch=train_loss_epoch,
                             val_loss_epoch=val_loss_epoch,
                             train_acc_epoch=train_acc_epoch,
                             val_acc_epoch=val_acc_epoch,
                             f1=f1,
                             val_precision=val_precision,
                             val_recall=val_recall,
                             train_mae=train_mae,
                             train_rmse=train_rmse,
                             val_mae=val_mae,
                             val_rmse=val_rmse)

        print("Обучение завершено!")
        print(f"Лучшая модель сохранена в '{self.new_folder_path}/best_model.pt' с val loss: {self.best_val_loss:.4f}")
        print(f"Последняя модель сохранена в '{self.new_folder_path}/last_model.pt'")
        
        return self.metrics