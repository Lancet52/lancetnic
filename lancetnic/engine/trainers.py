import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import (f1_score, 
                             precision_score, 
                             recall_score,
                             mean_absolute_error,
                             mean_squared_error,
                             hamming_loss, accuracy_score)
from tqdm import tqdm
from lancetnic.utils import Metrics


# =====================================================================
# БАЗОВЫЙ КЛАСС ТРЕНЕРА 
# =====================================================================

class BaseTrainer:
    """Базовый класс для всех тренеров"""
    
    def __init__(self, model, optimizer, device, train_loader, val_loader, new_folder_path, **extra):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.new_folder_path = new_folder_path
        self.best_val_loss = float('inf')
        self.metrics = {}
        self.mtx = Metrics()
        self.visualize = True

        for k, v in extra.items():
            setattr(self, k, v)
    
    
    def _train_one_epoch(self, epoch, num_epochs):
        raise NotImplementedError
    
    def _validate_one_epoch(self):
        raise NotImplementedError
    
    def _calculate_metrics(self, train_data, val_data):
        raise NotImplementedError
    
    def _get_csv_row(self, epoch, metrics):
        raise NotImplementedError
    
    def _get_payload(self, epoch, val_loss, **extra):
        raise NotImplementedError
    
    def _log_epoch(self, epoch, num_epochs, metrics):
        raise NotImplementedError
    
    # ---- Общие методы ----
    
    def save_hyperparameters(self, hyperparams):
        path_config = f"{self.new_folder_path}/hyperparams.yaml"
        with open(path_config, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
        print(f"Гиперпараметры сохранены в: {path_config}")
    
    def save_model(self, epoch, val_loss, payload):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.visualize:
                self._on_best_model(payload)
                torch.save(payload, f"{self.new_folder_path}/best_model.pt")
        if self.visualize:
            torch.save(payload, f"{self.new_folder_path}/last_model.pt")
    
    def _on_best_model(self, payload):
        """Вызывается при сохранении лучшей модели. Можно переопределить."""
        pass

    def _to_original_scale(self, values) -> np.ndarray:
        """Возвращает значения регрессии в исходные единицы"""
        arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        scaler = getattr(self, "scaler_reg", None)
        if scaler is not None:
            arr = scaler.inverse_transform(arr)
        return arr.ravel()
    
    def save_to_csv(self, csv_row):
        if not self.visualize:
            return
        csv_path = f"{self.new_folder_path}/result.csv"
        pd.DataFrame([csv_row]).to_csv(csv_path, mode='a', header=False, index=False)
    
    def train(self,
              num_epochs,
              hidden_size,
              num_layers,
              input_size,
              train_path,
              label_column,
              dropout,
              batch_size,
              learning_rate,
              optim_name,
              crit_name,
              **extra_model_params):
        """Главный цикл обучения. Работает для ВСЕХ задач."""
        hyperparams = {
            "model_params": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                **extra_model_params
            },
            "train_params": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optim_name,
                "criterion": crit_name
            }
        }
        if self.visualize:
            self.save_hyperparameters(hyperparams)
        
        for epoch in range(num_epochs):
            # Обучение
            train_data = self._train_one_epoch(epoch, num_epochs)
            # Валидация
            val_data = self._validate_one_epoch()
            # Метрики
            metrics = self._calculate_metrics(train_data, val_data)
            metrics['epoch'] = epoch + 1
            metrics['val_loss_epoch'] = metrics.get('val_loss_epoch', 0)
            
            # Сохранение модели
            payload = self._get_payload(epoch=epoch,
                                        val_loss=metrics['val_loss_epoch'],
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        input_size=input_size,
                                        **extra_model_params)
            
            self.save_model(epoch, metrics['val_loss_epoch'], payload)
            
            # Логирование и CSV
            if self.visualize:
                self._log_epoch(epoch, num_epochs, metrics)
                self.save_to_csv(self._get_csv_row(epoch, metrics))
        if self.visualize:
            print("Обучение завершено!")
            print(f"Лучшая модель сохранена в '{self.new_folder_path}\\best_model.pt' с val loss: {self.best_val_loss:.4f}")
            print(f"Последняя модель сохранена в '{self.new_folder_path}\\last_model.pt'")
        return self.metrics


# =====================================================================
# ТРЕНЕР КЛАССИФИКАЦИИ
# =====================================================================

class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, device,
                 train_loader, val_loader, label_encoder,
                 vectorizer_text, vectorizer_scalar, new_folder_path,
                 visualize=True):
        super().__init__(model, optimizer, device, train_loader, val_loader, new_folder_path,
                         criterion=criterion, label_encoder=label_encoder,
                         vectorizer_text=vectorizer_text, vectorizer_scalar=vectorizer_scalar,
                         visualize=visualize)
        self.metrics = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'f1_score': [],
            'all_preds': [], 'all_labels': []
        }
    
    def _train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train") if self.visualize else self.train_loader
        for inputs, labels in iterator:
            inputs, labels = inputs.float().to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return {'loss': loss_sum, 'correct': correct, 'total': total}
    
    def _validate_one_epoch(self):
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        iterator = tqdm(self.val_loader, desc="Validation") if self.visualize else self.val_loader
        with torch.no_grad():
            for inputs, labels in iterator:
                inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return {'loss': loss_sum, 'correct': correct, 'total': total,
                'preds': all_preds, 'labels': all_labels}
    
    def _calculate_metrics(self, train_data, val_data):
        tr_loss = train_data['loss'] / len(self.train_loader)
        val_loss = val_data['loss'] / len(self.val_loader)
        tr_acc = train_data['correct'] / train_data['total']
        val_acc = val_data['correct'] / val_data['total']
        true = np.concatenate(val_data['labels']).ravel()
        pred = np.concatenate(val_data['preds']).ravel()
        f1 = f1_score(true, pred, average='weighted', zero_division=0)
        prec = precision_score(true, pred, average='weighted', zero_division=0)
        rec = recall_score(true, pred, average='weighted', zero_division=0)
        
        self.metrics['epoch'].append(train_data.get('epoch', len(self.metrics['epoch']) + 1))
        self.metrics['train_loss'].append(tr_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(tr_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['f1_score'].append(f1)
        self.metrics['val_precision'].append(prec)
        self.metrics['val_recall'].append(rec)
        self.metrics['all_preds'].append(pred)
        self.metrics['all_labels'].append(true)
        
        return {
            'train_loss_epoch': tr_loss, 'val_loss_epoch': val_loss,
            'train_acc_epoch': tr_acc, 'val_acc_epoch': val_acc,
            'f1': f1, 'val_precision': prec, 'val_recall': rec
        }
    
    def _get_payload(self, epoch, val_loss, hidden_size, num_layers, input_size, num_classes, **_):
        return {
            'model': self.model, 'input_size': input_size,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'num_classes': num_classes,
            'vectorizer_text': self.vectorizer_text,
            'vectorizer_scalar': self.vectorizer_scalar,
            'label_encoder': self.label_encoder,
            'epoch': epoch, 'val_loss': val_loss,
            'val_acc': self.metrics['val_acc'][-1]
        }
    
    def _on_best_model(self, payload):
        self.mtx.confus_matrix(last_labels=self.metrics['all_labels'][-1],
                               last_preds=self.metrics['all_preds'][-1],
                               label_encoder=self.label_encoder.classes_,
                               save_folder_path=self.new_folder_path,
                               plt_name="confusion_matrix_best_model")
    
    def _get_csv_row(self, epoch, m):
        return {
            "epoch": epoch + 1,
            "train_loss": f"{m['train_loss_epoch']:.4f}",
            "train_acc, %": f"{100 * m['train_acc_epoch']:.2f}",
            "val_loss": f"{m['val_loss_epoch']:.4f}",
            "val_acc, %": f"{100 * m['val_acc_epoch']:.2f}",
            "val_precision, %": f"{100 * m['val_precision']:.2f}",
            "val_recall, %": f"{100 * m['val_recall']:.2f}",
            "F1_score": f"{100 * m['f1']:.2f}"
        }
    
    def _log_epoch(self, epoch, num_epochs, m):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {m['train_loss_epoch']:.4f} | Train Acc: {100 * m['train_acc_epoch']:.2f}%")
        print(f"Val Loss: {m['val_loss_epoch']:.4f} | Val Acc: {100 * m['val_acc_epoch']:.2f}% | "
              f"Val Prec: {100 * m['val_precision']:.2f}% | Val Rec: {100 * m['val_recall']:.2f}% | "
              f"F1: {100 * m['f1']:.2f}%")
        print("-" * 50)


# =====================================================================
# ТРЕНЕР РЕГРЕССИИ
# =====================================================================

class RegressionTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 device,
                 train_loader, 
                 val_loader,
                 vectorizer_text, 
                 vectorizer_scalar, 
                 new_folder_path,
                 scaler_reg=None):
        
        super().__init__(model, 
                         optimizer, 
                         device, 
                         train_loader, 
                         val_loader, 
                         new_folder_path,
                         criterion=criterion,
                         vectorizer_text=vectorizer_text, 
                         vectorizer_scalar=vectorizer_scalar,
                         scaler_reg=scaler_reg)
        self.metrics = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [],
            'all_val_preds': [], 'all_val_labels': []
        }
    
    def _train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        loss_sum = 0.0
        preds, labels = [], []
        iterator = tqdm(self.train_loader, desc="Training") if self.visualize else self.train_loader
        for inputs, labs in iterator:
            inputs, labs = inputs.float().to(self.device), labs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labs)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            preds.extend(outputs.cpu().detach().numpy())
            labels.extend(labs.cpu().numpy())
        return {'loss': loss_sum, 'preds': preds, 'labels': labels}
    
    def _validate_one_epoch(self):
        self.model.eval()
        loss_sum = 0.0
        preds, labels = [], []
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc="Validation") if self.visualize else self.val_loader
            for inputs, labs in iterator:
                inputs, labs = inputs.float().to(self.device), labs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labs)
                loss_sum += loss.item()
                preds.extend(outputs.cpu().numpy().flatten())
                labels.extend(labs.cpu().numpy().flatten())
        return {'loss': loss_sum, 'preds': preds, 'labels': labels}
    
    def _calculate_metrics(self, train_data, val_data):
        tr_loss = train_data['loss'] / len(self.train_loader)
        val_loss = val_data['loss'] / len(self.val_loader)
        tr_true = self._to_original_scale(train_data['labels'])
        tr_pred = self._to_original_scale(train_data['preds'])
        val_true = self._to_original_scale(val_data['labels'])
        val_pred = self._to_original_scale(val_data['preds'])
        tr_mae = mean_absolute_error(tr_true, tr_pred)
        val_mae = mean_absolute_error(val_true, val_pred)
        tr_rmse = np.sqrt(mean_squared_error(tr_true, tr_pred))
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        
        self.metrics['epoch'].append(len(self.metrics['epoch']) + 1)
        self.metrics['train_loss'].append(tr_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_mae'].append(tr_mae)
        self.metrics['val_mae'].append(val_mae)
        self.metrics['train_rmse'].append(tr_rmse)
        self.metrics['val_rmse'].append(val_rmse)
        self.metrics['all_val_preds'].append(val_pred)
        self.metrics['all_val_labels'].append(val_true)
        
        return {
            'train_loss_epoch': tr_loss, 'val_loss_epoch': val_loss,
            'train_mae': tr_mae, 'val_mae': val_mae,
            'train_rmse': tr_rmse, 'val_rmse': val_rmse
        }
    
    def _get_payload(self, epoch, val_loss, hidden_size, num_layers, input_size, output_size, **_):
        return {
            'model': self.model, 'input_size': input_size,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'output_size': output_size,
            'vectorizer_text': self.vectorizer_text,
            'vectorizer_scalar': self.vectorizer_scalar,
            'scaler_reg': self.scaler_reg,
            'epoch': epoch, 'val_loss': val_loss
        }
    
    def _get_csv_row(self, epoch, m):
        return {
            "epoch": epoch + 1,
            "train_loss": f"{m['train_loss_epoch']:.4f}",
            "val_loss": f"{m['val_loss_epoch']:.4f}",
            "train_mae": f"{m['train_mae']:.4f}",
            "val_mae": f"{m['val_mae']:.4f}",
            "train_rmse": f"{m['train_rmse']:.4f}",
            "val_rmse": f"{m['val_rmse']:.4f}"
        }
    
    def _log_epoch(self, epoch, num_epochs, m):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {m['train_loss_epoch']:.4f} | Train MAE: {m['train_mae']:.4f}")
        print(f"Val Loss: {m['val_loss_epoch']:.4f} | Val MAE: {m['val_mae']:.4f}")
        print("-" * 50)


# =====================================================================
# ТРЕНЕР МУЛЬТИЗАДАЧИ
# =====================================================================

class MultiTaskTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 criterion_class, 
                 criterion_reg, 
                 optimizer, 
                 device,
                 train_loader, 
                 val_loader, 
                 label_encoder,
                 vectorizer_text, 
                 vectorizer_scalar, 
                 new_folder_path,
                 loss_ratio_cls=1.0, 
                 loss_ratio_reg=1.0, 
                 scaler_reg=None):
        
        super().__init__(model, 
                         optimizer, 
                         device, 
                         train_loader, 
                         val_loader, 
                         new_folder_path,
                         criterion_class=criterion_class, 
                         criterion_reg=criterion_reg,
                         label_encoder=label_encoder,
                         vectorizer_text=vectorizer_text, 
                         vectorizer_scalar=vectorizer_scalar,
                         loss_ratio_cls=loss_ratio_cls, 
                         loss_ratio_reg=loss_ratio_reg,
                         scaler_reg=scaler_reg)
        
        self.metrics = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'f1_score': [],
            'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [],
            'all_preds': [], 'all_labels': [],
            'all_val_preds': [], 'all_val_labels': []
        }
    
    def _train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        reg_preds, reg_labels = [], []
        iterator = tqdm(self.train_loader, desc="Training") if self.visualize else self.train_loader
        for inputs, labels_cls, labels_reg in iterator:
            inputs = inputs.float().to(self.device)
            labels_cls, labels_reg = labels_cls.to(self.device), labels_reg.to(self.device)
            self.optimizer.zero_grad()
            class_out, reg_out = self.model(inputs)
            loss_c = self.criterion_class(class_out, labels_cls)
            loss_r = self.criterion_reg(reg_out, labels_reg)
            loss = self.loss_ratio_cls * loss_c + self.loss_ratio_reg * loss_r
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            _, predicted = torch.max(class_out.data, 1)
            total += labels_cls.size(0)
            correct += (predicted == labels_cls).sum().item()
            reg_preds.extend(reg_out.cpu().detach().numpy().flatten())
            reg_labels.extend(labels_reg.cpu().numpy().flatten())
        
        reg_true = self._to_original_scale(reg_labels)
        reg_pred = self._to_original_scale(reg_preds)
        tr_mae = mean_absolute_error(reg_true, reg_pred)
        tr_rmse = np.sqrt(mean_squared_error(reg_true, reg_pred))
        return {'loss': loss_sum, 'correct': correct, 'total': total,
                'reg_preds': reg_preds, 'reg_labels': reg_labels,
                'reg_mae': tr_mae, 'reg_rmse': tr_rmse}
    
    def _validate_one_epoch(self):
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        cls_preds, cls_labels = [], []
        reg_preds, reg_labels = [], []
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc="Validation") if self.visualize else self.val_loader
            for inputs, labels_cls, labels_reg in iterator:
                inputs = inputs.float().to(self.device)
                labels_cls, labels_reg = labels_cls.to(self.device), labels_reg.to(self.device)
                class_out, reg_out = self.model(inputs)
                loss_c = self.criterion_class(class_out, labels_cls)
                loss_r = self.criterion_reg(reg_out, labels_reg)
                loss_sum += (self.loss_ratio_cls * loss_c + self.loss_ratio_reg * loss_r).item()
                _, predicted = torch.max(class_out.data, 1)
                total += labels_cls.size(0)
                correct += (predicted == labels_cls).sum().item()
                cls_preds.extend(predicted.cpu().numpy())
                cls_labels.extend(labels_cls.cpu().numpy())
                reg_preds.extend(reg_out.cpu().numpy().flatten())
                reg_labels.extend(labels_reg.cpu().numpy().flatten())
        return {'loss': loss_sum, 'correct': correct, 'total': total,
                'cls_preds': cls_preds, 'cls_labels': cls_labels,
                'reg_preds': reg_preds, 'reg_labels': reg_labels}
    
    def _calculate_metrics(self, train_data, val_data):
        tr_loss = train_data['loss'] / len(self.train_loader)
        val_loss = val_data['loss'] / len(self.val_loader)
        tr_acc = train_data['correct'] / train_data['total']
        val_acc = val_data['correct'] / val_data['total']
        
        true_cls = np.array(val_data['cls_labels'])
        pred_cls = np.array(val_data['cls_preds'])
        f1 = f1_score(true_cls, pred_cls, average='weighted', zero_division=0)
        prec = precision_score(true_cls, pred_cls, average='weighted', zero_division=0)
        rec = recall_score(true_cls, pred_cls, average='weighted', zero_division=0)
        
        reg_true = self._to_original_scale(val_data['reg_labels'])
        reg_pred = self._to_original_scale(val_data['reg_preds'])
        val_mae = mean_absolute_error(reg_true, reg_pred)
        val_rmse = np.sqrt(mean_squared_error(reg_true, reg_pred))
        
        self.metrics['epoch'].append(len(self.metrics['epoch']) + 1)
        self.metrics['train_loss'].append(tr_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(tr_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['f1_score'].append(f1)
        self.metrics['val_precision'].append(prec)
        self.metrics['val_recall'].append(rec)
        self.metrics['train_mae'].append(train_data['reg_mae'])
        self.metrics['val_mae'].append(val_mae)
        self.metrics['train_rmse'].append(train_data['reg_rmse'])
        self.metrics['val_rmse'].append(val_rmse)
        self.metrics['all_preds'].append(pred_cls)
        self.metrics['all_labels'].append(true_cls)
        self.metrics['all_val_preds'].append(reg_pred)
        self.metrics['all_val_labels'].append(reg_true)
        
        return {
            'train_loss_epoch': tr_loss, 'val_loss_epoch': val_loss,
            'train_acc_epoch': tr_acc, 'val_acc_epoch': val_acc,
            'f1': f1, 'val_precision': prec, 'val_recall': rec,
            'train_mae': train_data['reg_mae'], 'val_mae': val_mae,
            'train_rmse': train_data['reg_rmse'], 'val_rmse': val_rmse
        }
    
    def _get_payload(self, epoch, val_loss, hidden_size, num_layers, input_size, num_classes, **_):
        return {
            'model': self.model, 'label_encoder': self.label_encoder,
            'vectorizer_text': self.vectorizer_text,
            'vectorizer_scalar': self.vectorizer_scalar,
            'scaler_reg': self.scaler_reg,
            'epoch': epoch, 'val_loss': val_loss,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'input_size': input_size, 'num_classes': num_classes
        }
    
    def _on_best_model(self, payload):
        self.mtx.confus_matrix(last_labels=self.metrics['all_labels'][-1],
                               last_preds=self.metrics['all_preds'][-1],
                               label_encoder=self.label_encoder.classes_,
                               save_folder_path=self.new_folder_path,
                               plt_name="confusion_matrix_best_model")
    
    def _get_csv_row(self, epoch, m):
        return {
            "epoch": epoch + 1,
            "train_loss": f"{m['train_loss_epoch']:.4f}",
            "val_loss": f"{m['val_loss_epoch']:.4f}",
            "train_acc, %": f"{100 * m['train_acc_epoch']:.2f}",
            "val_acc, %": f"{100 * m['val_acc_epoch']:.2f}",
            "val_precision, %": f"{100 * m['val_precision']:.2f}",
            "val_recall, %": f"{100 * m['val_recall']:.2f}",
            "F1_score": f"{100 * m['f1']:.2f}",
            "train_mae": f"{m['train_mae']:.4f}",
            "val_mae": f"{m['val_mae']:.4f}",
            "train_rmse": f"{m['train_rmse']:.4f}",
            "val_rmse": f"{m['val_rmse']:.4f}"
        }
    
    def _log_epoch(self, epoch, num_epochs, m):
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"TrLoss: {m['train_loss_epoch']:.4f} | ValLoss: {m['val_loss_epoch']:.4f} | "
              f"ValAcc: {m['val_acc_epoch']*100:.2f}% | F1: {m['f1']*100:.2f}% | "
              f"ValMAE: {m['val_mae']:.4f} | ValRMSE: {m['val_rmse']:.4f}")
        print("-" * 50)


# =====================================================================
# ТРЕНЕР МУЛЬТИЛЕЙБЛ КЛАССИФИКАЦИИ
# =====================================================================

class MultiLabelTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 device,
                 train_loader, 
                 val_loader, 
                 label_binarizer,
                 vectorizer_text, 
                 vectorizer_scalar, 
                 new_folder_path,
                 threshold=0.5):
        
        super().__init__(model, 
                         optimizer, 
                         device, 
                         train_loader, 
                         val_loader, 
                         new_folder_path,
                         criterion=criterion, 
                         label_binarizer=label_binarizer,
                         vectorizer_text=vectorizer_text, 
                         vectorizer_scalar=vectorizer_scalar)
        
        self.threshold = threshold
        self.metrics = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_subset_acc': [], 'val_subset_acc': [],
            'val_hamming_loss': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'all_preds': [], 'all_labels': [], 'all_probs': []
        }
    
    def _train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        loss_sum = 0.0
        preds, labels = [], []
        iterator = tqdm(self.train_loader, desc="Training") if self.visualize else self.train_loader
        for inputs, labs in iterator:
            inputs = inputs.float().to(self.device)
            labs = labs.float().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labs)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            p = (torch.sigmoid(outputs) >= self.threshold).int()
            preds.append(p.cpu().numpy())
            labels.append(labs.cpu().numpy())
        return {'loss': loss_sum, 'preds': preds, 'labels': labels}
    
    def _validate_one_epoch(self):
        self.model.eval()
        loss_sum = 0.0
        preds, labels, probs = [], [], []
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc="Validation") if self.visualize else self.val_loader
            for inputs, labs in iterator:
                inputs = inputs.float().to(self.device)
                labs = labs.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labs)
                loss_sum += loss.item()
                p = torch.sigmoid(outputs)
                pr = (p >= self.threshold).int()
                probs.append(p.cpu().numpy())
                preds.append(pr.cpu().numpy())
                labels.append(labs.cpu().numpy())
        return {'loss': loss_sum, 'preds': preds, 'labels': labels, 'probs': probs}
    
    def _calculate_metrics(self, train_data, val_data):
        tr_loss = train_data['loss'] / len(self.train_loader)
        val_loss = val_data['loss'] / len(self.val_loader)
        
        y_tr_true = np.vstack(train_data['labels'])
        y_tr_pred = np.vstack(train_data['preds'])
        y_val_true = np.vstack(val_data['labels'])
        y_val_pred = np.vstack(val_data['preds'])
        
        tr_acc = accuracy_score(y_tr_true, y_tr_pred)
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_hamming = hamming_loss(y_val_true, y_val_pred)
        val_prec = precision_score(y_val_true, y_val_pred, average='samples', zero_division=0)
        val_rec = recall_score(y_val_true, y_val_pred, average='samples', zero_division=0)
        val_f1 = f1_score(y_val_true, y_val_pred, average='samples', zero_division=0)
        
        self.metrics['epoch'].append(len(self.metrics['epoch']) + 1)
        self.metrics['train_loss'].append(tr_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_subset_acc'].append(tr_acc)
        self.metrics['val_subset_acc'].append(val_acc)
        self.metrics['val_hamming_loss'].append(val_hamming)
        self.metrics['val_precision'].append(val_prec)
        self.metrics['val_recall'].append(val_rec)
        self.metrics['val_f1'].append(val_f1)
        self.metrics['all_preds'].append(np.vstack(val_data['preds']))
        self.metrics['all_labels'].append(np.vstack(val_data['labels']))
        self.metrics['all_probs'].append(np.vstack(val_data['probs']))
        
        return {
            'train_loss_epoch': tr_loss, 'val_loss_epoch': val_loss,
            'train_subset_acc': tr_acc, 'val_subset_acc': val_acc,
            'val_hamming': val_hamming,
            'val_precision': val_prec, 'val_recall': val_rec, 'val_f1': val_f1
        }
    
    def _get_payload(self, epoch, val_loss, hidden_size, num_layers, input_size, num_classes, **_):
        return {
            'model': self.model, 'input_size': input_size,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'num_classes': num_classes,
            'vectorizer_text': self.vectorizer_text,
            'vectorizer_scalar': self.vectorizer_scalar,
            'label_binarizer': self.label_binarizer,
            'threshold': self.threshold,
            'epoch': epoch, 'val_loss': val_loss
        }
    
    def _on_best_model(self, payload):
        self.mtx.multi_label_confusion_matrix(last_labels=self.metrics['all_labels'][-1],
                                              last_probs=self.metrics['all_probs'][-1],
                                              label_binarizer=self.label_binarizer,
                                              save_folder_path=self.new_folder_path,
                                              plt_name="confusion_matrix_best_model",
                                              threshold=self.threshold)
    
    def _get_csv_row(self, epoch, m):
        return {
            "epoch": epoch + 1,
            "train_loss": f"{m['train_loss_epoch']:.4f}",
            "val_loss": f"{m['val_loss_epoch']:.4f}",
            "train_subset_acc, %": f"{100 * m['train_subset_acc']:.2f}",
            "val_subset_acc, %": f"{100 * m['val_subset_acc']:.2f}",
            "val_hamming_loss": f"{m['val_hamming']:.4f}",
            "val_precision, %": f"{100 * m['val_precision']:.2f}",
            "val_recall, %": f"{100 * m['val_recall']:.2f}",
            "val_f1, %": f"{100 * m['val_f1']:.2f}"
        }
    
    def _log_epoch(self, epoch, num_epochs, m):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {m['train_loss_epoch']:.4f} | "
              f"Train SubsetAcc: {100 * m['train_subset_acc']:.2f}%")
        print(f"Val Loss: {m['val_loss_epoch']:.4f} | "
              f"Val SubsetAcc: {100 * m['val_subset_acc']:.2f}% | "
              f"Hamming: {m['val_hamming']:.4f} | "
              f"Prec: {100 * m['val_precision']:.2f}% | "
              f"Rec: {100 * m['val_recall']:.2f}% | "
              f"F1: {100 * m['val_f1']:.2f}%")
        print("-" * 60)