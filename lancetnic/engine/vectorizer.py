import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Векторизация тектовых данных
def vectorize_text(text_column, df_train, max_features):
    if isinstance(text_column, str):
        text_column = [text_column]
    text_encoder_list = []
    vectorizers = []
    for text_col in text_column:
        vectorizer_text = TfidfVectorizer(max_features=max_features)
        text_encoder = vectorizer_text.fit_transform(df_train[text_col].fillna('')).toarray()            
        text_encoder_list.append(text_encoder)
        vectorizers.append(vectorizer_text)

    combined_text = np.hstack(text_encoder_list)    
    return combined_text, vectorizers


# Векторизация числовых данных с отдельным валидационным датасетом
def vectorize_text_val(text_column, df_train, df_val, max_features):
    if isinstance(text_column, str):
        text_column = [text_column]
            
    vectorizer_text = []
    text_encoder_list_train = []
    text_encoder_list_val = []
            
    for text_col in text_column:
        vectorizer_textdata = TfidfVectorizer(max_features=max_features)
                
        # Обработка train данных
        text_encoder_train = vectorizer_textdata.fit_transform(df_train[text_col].fillna('').astype(str)).toarray()
        # Обработка val данных
        text_encoder_val = vectorizer_textdata.transform(df_val[text_col].fillna('').astype(str)).toarray()
                
        text_encoder_list_train.append(text_encoder_train)
        text_encoder_list_val.append(text_encoder_val)
        vectorizer_text.append(vectorizer_textdata)
            
    X_train = np.hstack(text_encoder_list_train)
    X_val = np.hstack(text_encoder_list_val)
    return X_train, X_val, vectorizer_text


# Векторизация числовых данных
def vectorize_data(data_column, df_train):
    if isinstance(data_column, str):
          data_column = [data_column]
    data_encoder_list = []
    vectorizers = []
    for data_col in data_column:
        vectorizer_scalar = StandardScaler()
        data_encoder = vectorizer_scalar.fit_transform(df_train[data_col].fillna(0).values.reshape(-1, 1))
        data_encoder_list.append(data_encoder)
        vectorizers.append(vectorizer_scalar)

    combined_data = np.hstack(data_encoder_list)   
    return combined_data, vectorizers

# Векторизация текстовых данных с отдельным валидационным датасетом
def vectorize_data_val(data_column, df_train, df_val):
    if isinstance(data_column, str):
        data_column = [data_column]
            
    vectorizer_scalar = []
    data_encoder_list_train = []
    data_encoder_list_val = []
            
    for data_col in data_column:
        vectorizer_data = StandardScaler()                
        # Обработка train данных
        data_encoder_train = vectorizer_data.fit_transform(df_train[data_col].fillna(0).values.reshape(-1, 1))
        # Обработка val данных
        data_encoder_val = vectorizer_data.transform(df_val[data_col].fillna(0).values.reshape(-1, 1))
                
        data_encoder_list_train.append(data_encoder_train)
        data_encoder_list_val.append(data_encoder_val)
        vectorizer_scalar.append(vectorizer_data)
            
    X_train = np.hstack(data_encoder_list_train)
    X_val = np.hstack(data_encoder_list_val)
    return X_train, X_val, vectorizer_scalar