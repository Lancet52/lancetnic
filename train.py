import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

class Trainer:
    def __init__(self, model, train_path, val_path, text_column='description', label_column='category'):
        self.model=model
        self.train_path=train_path
        self.val_path=val_path
        self.text_column=text_column
        self.label_column=label_column
        
        self.df_train=None
        self.df_val=None
        self.vectorizer=None
        self.X_train=None
        self.X_val=None
        self.label_encoder=None
        self.y_train=None
        self.y_val=None
        self.input_size=None
        self.num_classes=None
        

    def load_data(self):
        """Загрузка и предобработка данных"""
        self.df_train=pd.read_csv(self.train_path)
        self.df_val=pd.read_csv(self.val_path)
        
        # Векторизация текста
        self.vectorizer=TfidfVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.df[self.text_column]).toarray()
        self.X_val = self.vectorizer.transform(self.df_val[self.text_column]).toarray()
        
        # Кодирование меток
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.df[self.label_column])
        self.y_val = self.label_encoder.transform(self.df_val[self.label_column])
        
        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
        
    def create_train_folder(self):
        """Создание папки для сохранения результатов обучения"""
        base_dir = os.getcwd()
        train_folders = []
        
        for folder in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder)) and folder.startswith("train_"):
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
        self.new_folder_path = os.path.join(base_dir, new_folder)
        
        os.makedirs(self.new_folder_path, exist_ok=True)
        print(f"Создана папка: {self.new_folder_path}")
        
        # Создание файла для результатов
        headers = ["epoch", "train_loss", "train_acc, %", 
                  "val_loss", "val_acc, %", "F1_score"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)