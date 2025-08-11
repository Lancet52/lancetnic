import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
        plt.savefig(f"{self.save_folder_path}/dataset_counts.png")
        plt.close()
