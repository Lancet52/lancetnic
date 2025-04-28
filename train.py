import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


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