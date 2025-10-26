import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error

class Metrics:
    # Базовый класс метрик
    def __init__(self, figsize=(10, 8), cmap="Blues"):
        self.figsize = figsize
        self.cmap = cmap

    def confus_matrix(self, last_labels, last_preds, label_encoder, save_folder_path, plt_name="confusion_matrix"):
        self.last_labels = last_labels
        self.last_preds = last_preds
        self.label_encoder = label_encoder
        self.save_folder_path = save_folder_path
        self.plt_name = plt_name
        self.conf_matrix = confusion_matrix(self.last_labels, self.last_preds)
        plt.figure(figsize=self.figsize)

        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap=self.cmap,
                    xticklabels=self.label_encoder,
                    yticklabels=self.label_encoder)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(f"{self.save_folder_path}/{self.plt_name}.png")
        plt.close()

    def train_val_loss(self, epoch, train_loss, val_loss, save_folder_path):
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.save_folder_path = save_folder_path
        plt.figure(figsize=self.figsize)
        plt.plot(self.epoch, self.train_loss, label='Train Loss')
        plt.plot(self.epoch, self.val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder_path}/train_val_loss.png")
        plt.close()

    def train_val_acc(self, epoch, train_acc, val_acc, save_folder_path):
        self.epoch = epoch
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.save_folder_path = save_folder_path

        plt.figure(figsize=self.figsize)
        plt.plot(self.epoch, self.train_acc, label='Train Accuracy')
        plt.plot(self.epoch, self.val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder_path}/train_val_acc.png")
        plt.close()

    def f1score(self, epoch, f1_score, save_folder_path):
        self.epoch = epoch
        self.f1_score = f1_score
        self.save_folder_path = save_folder_path

        plt.figure(figsize=self.figsize)
        plt.plot(self.epoch, self.f1_score,
                 label='Validation F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.title('F1-score during Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder_path}/f1_score.png")
        plt.close()

    def dataset_counts(self, data_path, label_column, save_folder_path):
        self.data_path = data_path
        self.save_folder_path = save_folder_path
        self.label_column = label_column

        df = pd.read_csv(self.data_path)

        counts = df[self.label_column].value_counts()
      
        plt.figure(figsize=self.figsize)
        plt.bar(counts.index, counts.values)
        plt.xlabel('Classes')
        plt.ylabel('Counts of classes')
        plt.title('Class distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.grid(axis='x', visible=False)
        plt.savefig(f"{self.save_folder_path}/dataset_counts.png")
        plt.close()

    # График Средней абсолютной ошибки (MAE)
    def train_val_mae(self, epoch, train_mae, val_mae, save_folder_path):
        self.epoch = epoch
        self.train_mae = train_mae
        self.val_mae = val_mae
        self.save_folder_path = save_folder_path

        plt.figure(figsize=self.figsize)
        plt.plot(self.epoch, self.train_mae, label='Train MAE')
        plt.plot(self.epoch, self.val_mae, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder_path}/train_val_mae.png")
        plt.close()

    # График RMSE по эпохам
    def train_val_rmse(self, epoch, train_rmse, val_rmse, save_folder_path):
        self.epoch = epoch
        self.train_rmse = train_rmse
        self.val_rmse = val_rmse
        self.save_folder_path = save_folder_path

        plt.figure(figsize=self.figsize)
        plt.plot(self.epoch, self.train_rmse, label='Train RMSE')
        plt.plot(self.epoch, self.val_rmse, label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder_path}/train_val_rmse.png")
        plt.close()

    # Точечный график (Scatter plot) и линия регрессии
    def regression_scatter(self, metrics, save_folder_path):
        """Scatter plot предсказаний vs реальных значений с линией регрессии"""
        self.save_folder_path = save_folder_path
        
        last_val_preds = metrics['all_val_preds'][-1]
        last_val_labels = metrics['all_val_labels'][-1]
        y_true = np.array(last_val_labels)
        y_pred = np.array(last_val_preds)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
        
        # Линия идеальных предсказаний (y = x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
        
        # Линия линейной регрессии
        if len(y_true) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
            line_x = np.array([min_val, max_val])
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, 'g-', linewidth=2, label=f'Regression line (R²={r_value**2:.3f})')
        
        plt.xlabel('Истинные значения')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values with Regression Line')
        plt.legend()
        plt.grid(True, alpha=0.3)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r_value**2 if len(y_true) > 1 else 0
        
        textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)        
        plt.tight_layout()
        plt.savefig(f"{self.save_folder_path}/regression_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Метрики регрессии - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")