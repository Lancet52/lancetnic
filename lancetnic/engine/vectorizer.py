import re
from collections import Counter
from collections.abc import Iterable
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Векторизация тектовых данных
def vectorize_text(text_column, df_train, max_features, text_mode='tfidf', max_len=100):
    if isinstance(text_column, str):
        text_column = [text_column]
    text_encoder_list = []
    vectorizers = []
    for text_col in text_column:
        vectorizer_text = make_text_vectorizer(text_mode, max_features, max_len)
        text_encoder = vectorizer_text.fit_transform(df_train[text_col].fillna(''))           
        text_encoder_list.append(text_encoder)
        vectorizers.append(vectorizer_text)

    combined_text = sp.hstack(text_encoder_list).tocsr() 
    return combined_text, vectorizers


# Векторизация числовых данных с отдельным валидационным датасетом
def vectorize_text_val(text_column, df_train, df_val, max_features, text_mode='tfidf', max_len=100):
    if isinstance(text_column, str):
        text_column = [text_column]
            
    vectorizer_text = []
    text_encoder_list_train = []
    text_encoder_list_val = []
            
    for text_col in text_column:
        vectorizer_textdata = make_text_vectorizer(text_mode, max_features, max_len)
                
        # Обработка train данных
        text_encoder_train = vectorizer_textdata.fit_transform(df_train[text_col].fillna('').astype(str))
        # Обработка val данных
        text_encoder_val = vectorizer_textdata.transform(df_val[text_col].fillna('').astype(str))
                
        text_encoder_list_train.append(text_encoder_train)
        text_encoder_list_val.append(text_encoder_val)
        vectorizer_text.append(vectorizer_textdata)
            
    X_train = sp.hstack(text_encoder_list_train).tocsr()
    X_val = sp.hstack(text_encoder_list_val).tocsr()
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

class SequenceVectorizer:
    PAD_ID: int = 0
    UNK_ID: int = 1
    TOKEN_PATTERN: str = r"(?u)\b\w\w+\b"

    def __init__(self, max_features: int | None = None, max_len: int = 100) -> None:
        self.max_features = max_features
        self.max_len = max_len
        self.vocabulary_: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary_) + 2

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(self.TOKEN_PATTERN, str(text).lower())

    def fit(self, texts: Iterable[str]) -> "SequenceVectorizer":
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        most_common = counter.most_common(self.max_features)
        self.vocabulary_ = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        return self

    def transform(self, texts: Iterable[str]) -> sp.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        ids: list[int] = []
        n_texts = 0
        for row, text in enumerate(texts):
            n_texts += 1
            tokens = self._tokenize(text)[:self.max_len]
            rows.extend([row] * len(tokens))
            cols.extend(range(len(tokens)))
            ids.extend(self.vocabulary_.get(tok, self.UNK_ID) for tok in tokens)
        return sp.csr_matrix((np.asarray(ids, dtype=np.float32), (rows, cols)),
                             shape=(n_texts, self.max_len))

    def fit_transform(self, texts: Iterable[str]) -> sp.csr_matrix:
        texts = list(texts)
        return self.fit(texts).transform(texts)

def make_text_vectorizer(text_mode: str,
                         max_features: int | None,
                         max_len: int) -> TfidfVectorizer | SequenceVectorizer:
    
    if text_mode == 'tfidf':
        return TfidfVectorizer(max_features=max_features)
    if text_mode == 'sequence':
        return SequenceVectorizer(max_features=max_features, max_len=max_len)
    raise ValueError(f"Неизвестный text_mode='{text_mode}'. Допустимо: 'tfidf' или 'sequence'")