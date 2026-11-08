import os
import torch
import yaml
import optuna
import warnings
from optuna.samplers import TPESampler
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

from lancetnic.utils import Metrics, dirruns
from lancetnic.engine.trainers import (
    ClassificationTrainer, RegressionTrainer,
    MultiTaskTrainer, MultiLabelTrainer
)
from lancetnic.engine.vectorizer import (
    vectorize_text, vectorize_data,
    vectorize_text_val, vectorize_data_val
)


# =====================================================================
# ДАТАСЕТЫ 
# =====================================================================

class ClassifierDataset(Dataset):
    def __init__(self, X, y):
        self.is_sparse = sparse.issparse(X)
        self.X = X
        if not self.is_sparse:
            self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        else:
            x = self.X[idx]
        return x, self.y[idx]


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.is_sparse = sparse.issparse(X)
        self.X = X
        if not self.is_sparse:
            self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        else:
            x = self.X[idx]
        return x, self.y[idx]


class MultiTaskDataset(Dataset):
    def __init__(self, X, y_class, y_reg):
        self.is_sparse = sparse.issparse(X)
        self.X = X
        if not self.is_sparse:
            self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        else:
            x = self.X[idx]
        return x, self.y_class[idx], self.y_reg[idx]


class MultiLabelDataset(Dataset):
    def __init__(self, X, y_multi_hot):
        self.is_sparse = sparse.issparse(X)
        self.X = X
        if not self.is_sparse:
            self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_multi_hot, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        else:
            x = self.X[idx]
        return x, self.y[idx]


# =====================================================================
# ФАБРИКИ 
# =====================================================================

OPTIMIZERS = {
    'Adam': optim.Adam, 'RAdam': optim.RAdam, 'SGD': optim.SGD,
    'RMSprop': optim.RMSprop, 'Adadelta': optim.Adadelta
}

CLASS_CRITERIA = {'CELoss': nn.CrossEntropyLoss}
REG_CRITERIA = {'MSELoss': nn.MSELoss, 'L1Loss': nn.L1Loss, 'SmoothL1Loss': nn.SmoothL1Loss}
MULTI_LABEL_CRITERIA = {'BCEWithLogitsLoss': nn.BCEWithLogitsLoss}

DEVICES = {
    'cuda': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'cpu': torch.device('cpu')
}


# =====================================================================
# ФУНКЦИИ
# =====================================================================
def _finetune_arch_params(checkpoint: dict,
                          model: nn.Module,
                          hidden_size: int | None,
                          num_layers: int | None,
                          dropout: float | None) -> tuple[int, int, float]:
    """Определяет реальные параметры архитектуры для дообучения"""
    saved_hidden = checkpoint.get('hidden_size')
    saved_layers = checkpoint.get('num_layers')

    for name, passed, saved in (("hidden_size", hidden_size, saved_hidden),
                                ("num_layers", num_layers, saved_layers)):
        if passed is not None and saved is not None and passed != saved:
            warnings.warn(
                f"{name}={passed} игнорируется: при дообучении архитектура "
                f"берётся из чекпоинта ({name}={saved})."
            )

    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if dropout is not None:
        for layer in dropout_layers:
            layer.p = dropout
    actual_dropout = dropout_layers[0].p if dropout_layers else 0.0
    real_hidden = saved_hidden if saved_hidden is not None else hidden_size
    real_layers = saved_layers if saved_layers is not None else num_layers
    return real_hidden, real_layers, actual_dropout

def _extract_features(text_column, data_column, df_train, df_val, max_features,
                      text_mode='tfidf', max_len=100):
    has_val = df_val is not None
    
    # --- Только числовые признаки ---
    if text_column is None and data_column is not None:
        if has_val:
            X_tr, X_val, vec_s = vectorize_data_val(data_column, df_train, df_val)
            return X_tr, X_val, None, vec_s
        else:
            X_all, vec_s = vectorize_data(data_column, df_train)
            return X_all, None, None, vec_s

    # --- Только текст ---
    elif data_column is None and text_column is not None:
        if has_val:
            X_tr, X_val, vec_t = vectorize_text_val(text_column, df_train, df_val, max_features, text_mode, max_len)
            return X_tr, X_val, vec_t, None
        else:
            X_all, vec_t = vectorize_text(text_column, df_train, max_features, text_mode, max_len)
            return X_all, None, vec_t, None

    # --- Текст + числовые признаки ---
    elif text_column is not None and data_column is not None:
        if has_val:
            txt_tr, txt_val, vec_t = vectorize_text_val(text_column, df_train, df_val, max_features, text_mode, max_len)
            dat_tr, dat_val, vec_s = vectorize_data_val(data_column, df_train, df_val)
            X_tr = sparse.hstack([txt_tr, sparse.csr_matrix(dat_tr)]).tocsr()
            X_val = sparse.hstack([txt_val, sparse.csr_matrix(dat_val)]).tocsr()
            return X_tr, X_val, vec_t, vec_s
        else:
            txt_all, vec_t = vectorize_text(text_column, df_train, max_features, text_mode, max_len)
            dat_all, vec_s = vectorize_data(data_column, df_train)
            X_all = sparse.hstack([txt_all, sparse.csr_matrix(dat_all)]).tocsr()
            return X_all, None, vec_t, vec_s

    else:
        raise ValueError("Должен быть указан хотя бы один из параметров: text_column или data_column")


def _prepare_inference_data(text_data, scalar_data, vec_text, vec_scalar):
    """
    УНИВЕРСАЛЬНАЯ подготовка данных для predict().
    """
    if text_data is None and scalar_data is None:
        raise ValueError("Должен быть передан хотя бы один из: text_data или scalar_data")

    # Нормализация векторайзеров в списки
    if not isinstance(vec_text, (list, tuple)):
        vec_text = [vec_text] if vec_text is not None else []
    if not isinstance(vec_scalar, (list, tuple)):
        vec_scalar = [vec_scalar] if vec_scalar is not None else []

    if text_data is not None and len(vec_text) == 1 and isinstance(text_data, str):
        text_data = [text_data]
    if scalar_data is not None and len(vec_scalar) == 1 and not isinstance(scalar_data, (list, tuple)):
        scalar_data = [scalar_data]

    if text_data is not None and len(text_data) != len(vec_text):
        raise ValueError(f"Ожидалось {len(vec_text)} текстовых колонок, получено {len(text_data)}")
    if scalar_data is not None and len(scalar_data) != len(vec_scalar):
        raise ValueError(f"Ожидалось {len(vec_scalar)} числовых колонок, получено {len(scalar_data)}")

    features = []
    if text_data is not None:
        for val, vec in zip(text_data, vec_text):
            features.append(vec.transform([val]).toarray())
    if scalar_data is not None:
        for val, vec in zip(scalar_data, vec_scalar):
            features.append(vec.transform([[val]]))

    X = np.hstack(features)
    return torch.tensor(X, dtype=torch.float32)

def _sequence_model_kwargs(text_mode: str,
                           vectorizer_text: list | None,
                           max_len: int) -> dict:
    """Собирает дополнительные параметры для моделей LSTM"""
    if text_mode != 'sequence':
        return {}
    if not vectorizer_text:
        raise ValueError("text_mode='sequence' требует хотя бы одну текстовую колонку (text_column)")
    vecs = vectorizer_text if isinstance(vectorizer_text, list) else [vectorizer_text]
    return {'vocab_sizes': [vec.vocab_size for vec in vecs],
            'max_len': max_len}
# =====================================================================
# БАЗОВЫЙ КЛАСС ДЛЯ КЛАССИФИКАЦИИ
# =====================================================================

class BaseClassificationPipeline:
    """
    Базовый класс для всех задач классификации.
    """
    
    # --- Параметры Dataset и Trainer ---
    _dataset_class = None
    _trainer_class = None
    _criteria_dict = None
    _default_criterion = None
    
    def __init__(self, text_column, data_column, label_column,
                 split_ratio, random_state, max_features,
                 text_mode='tfidf', max_len=100):
        self.text_column = text_column
        self.data_column = data_column
        self.label_column = label_column
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.max_features = max_features
        self.text_mode = text_mode
        self.max_len = max_len
        
        # Для пайплайна
        self.df_train = None
        self.df_val = None
        self.vectorizer_text = None
        self.vectorizer_scalar = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.input_size = None
        self.num_classes = None
        self.model = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.metrics = None
        self.new_folder_path = None
        self.csv_path = None
        self.mtx = Metrics()
    
    
    def _encode_labels(self):
        """Кодирование меток"""
        raise NotImplementedError
    
    def _get_csv_headers(self):
        """Заголовки для CSV"""
        raise NotImplementedError
    
    def _visualize_metrics(self, metrics):
        """Визуализация метрик"""
        raise NotImplementedError
    
    # ---- Общая логика ----
    
    def _vectorize_features(self, df_val):
        """Векторизация признаков"""
        X_tr, X_val, vec_t, vec_s = _extract_features(
            self.text_column, self.data_column,
            self.df_train, df_val, self.max_features,
            self.text_mode, self.max_len
        )
        self.vectorizer_text = vec_t
        self.vectorizer_scalar = vec_s
        return X_tr, X_val
    
    def vectorize_with_val_path(self):
        """Векторизация с отдельным валидационным набором"""
        self.df_val = pd.read_csv(self.val_path)
        X_tr, X_val = self._vectorize_features(self.df_val)
        self.X_train, self.X_val = X_tr, X_val
        self._encode_labels()
        self.input_size = self.X_train.shape[1]
        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes
    
    def vectorize_no_val_path(self) -> tuple:
        """Векторизация без отдельного валидационного набора"""
        self.df_train, self.df_val = train_test_split(self.df_train, test_size=self.split_ratio, random_state=self.random_state)
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_val = self.df_val.reset_index(drop=True)

        X_tr, X_val = self._vectorize_features(self.df_val)
        self.X_train, self.X_val = X_tr, X_val
        self._encode_labels()
        self.input_size = self.X_train.shape[1]
        return self.X_train, self.X_val, self.y_train, self.y_val, self.input_size, self.num_classes
    
    def train(self, 
              model_name, 
              train_path, 
              val_path, 
              num_epochs,
              hidden_size=256, 
              num_layers=1, 
              batch_size=128, 
              learning_rate=0.001,
              dropout=0, 
              optim_name='Adam', 
              crit_name=None, 
              device_name='cuda',
              folder_name='train_',
              visualize=True,
              save_folder_path=None,
              **trainer_kwargs):
        """Общий метод обучения для всех классификаторов"""
        self.train_path = train_path
        self.val_path = val_path
        self.df_train = pd.read_csv(train_path)
        
        # Векторизация
        if val_path is None:
            self.vectorize_no_val_path()
        else:
            self.vectorize_with_val_path()
        
        if save_folder_path is None:
            self.new_folder_path = dirruns(folder_name)
        else:
            self.new_folder_path = save_folder_path
        
        # CSV с результатами
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if visualize and not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=self._get_csv_headers()).to_csv(self.csv_path, index=False)
        
        # DataLoader
        train_dataset = self._dataset_class(self.X_train, self.y_train)
        val_dataset = self._dataset_class(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Модель
        self.model = model_name(self.input_size, hidden_size, num_layers, self.num_classes, dropout,
                                **_sequence_model_kwargs(self.text_mode, self.vectorizer_text, self.max_len))
        crit_name = crit_name or self._default_criterion
        criterion = self._criteria_dict.get(crit_name, nn.CrossEntropyLoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        if visualize:
            print(f"Используется устройство: {device}")
        
        # Тренер
        trainer = self._trainer_class(model=self.model, 
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      device=device,
                                      train_loader=train_loader, 
                                      val_loader=val_loader,
                                      vectorizer_text=self.vectorizer_text, 
                                      vectorizer_scalar=self.vectorizer_scalar,
                                      new_folder_path=self.new_folder_path,
                                      **self._get_trainer_kwargs(), 
                                      **trainer_kwargs)
        trainer.visualize = visualize
        # Запуск обучения
        metrics = trainer.train(num_epochs=num_epochs, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers,
                                input_size=self.input_size, 
                                num_classes=self.num_classes,
                                train_path=train_path, 
                                label_column=self.label_column,
                                dropout=dropout, 
                                batch_size=batch_size, 
                                learning_rate=learning_rate,
                                optim_name=optim_name, 
                                crit_name=crit_name)
        
        if visualize:
            self._visualize_metrics(metrics)
            return None
        return metrics
    
    def _get_trainer_kwargs(self):
        """Дополнительные параметры для конструктора тренера"""
        return {}
    
    def predict(self, model_path, text_data=None, scalar_data=None):
        """Базовый predict"""
        X = _prepare_inference_data(text_data, 
                                    scalar_data,
                                    None, 
                                    None)

        raise NotImplementedError("Используйте predict в наследниках")


# =====================================================================
# КЛАССЫ-НАСЛЕДНИКИ
# =====================================================================

class Classification(BaseClassificationPipeline):
    """Обычная классификация (один класс на объект)."""
    
    _dataset_class = ClassifierDataset
    _trainer_class = ClassificationTrainer
    _criteria_dict = CLASS_CRITERIA
    _default_criterion = 'CELoss'
    
    def __init__(self, 
                 text_column=None, 
                 data_column=None, 
                 label_column=None,
                 split_ratio=0.2, 
                 random_state=42, 
                 max_features=None,
                 text_mode='tfidf',
                 max_len=100):
        
        super().__init__(text_column, 
                         data_column, 
                         label_column,
                         split_ratio, 
                         random_state, 
                         max_features,
                         text_mode=text_mode,
                         max_len=max_len)
        
        self.label_encoder = None
    
    def _encode_labels(self) -> None:
        """Кодирует названия классов в числа"""
        self.label_encoder = LabelEncoder()
        all_labels = pd.concat([self.df_train[self.label_column],
                                self.df_val[self.label_column]])
        self.label_encoder.fit(all_labels)
        self.y_train = self.label_encoder.transform(self.df_train[self.label_column])
        self.y_val = self.label_encoder.transform(self.df_val[self.label_column])
        self.num_classes = len(self.label_encoder.classes_)
    
    def _get_trainer_kwargs(self):
        return {'label_encoder': self.label_encoder}
    
    def _get_csv_headers(self):
        return ["epoch", "train_loss", "train_acc, %", "val_loss", "val_acc, %",
                "val_precision, %", "val_recall,%", "F1_score"]
    
    def _visualize_metrics(self, metrics):
        self.mtx.confus_matrix(last_labels=metrics['all_labels'][-1], 
                               last_preds=metrics['all_preds'][-1],
                               label_encoder=self.label_encoder.classes_,
                               save_folder_path=self.new_folder_path, 
                               plt_name="confusion_matrix_last_model")
        
        self.mtx.train_val_loss(epoch=metrics['epoch'],
                                train_loss=metrics['train_loss'],
                                val_loss=metrics['val_loss'], 
                                save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_acc(epoch=metrics['epoch'], 
                               train_acc=metrics['train_acc'],
                               val_acc=metrics['val_acc'], 
                               save_folder_path=self.new_folder_path)
        
        self.mtx.f1score(epoch=metrics['epoch'], 
                         f1_score=metrics['f1_score'],
                         save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_precision(epoch=metrics['epoch'], 
                                     val_precision=metrics['val_precision'],
                                     save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_recall(epoch=metrics['epoch'], 
                                  val_recall=metrics['val_recall'],
                                  save_folder_path=self.new_folder_path)
        
        self.mtx.dataset_counts(data_path=self.train_path, 
                                label_column=self.label_column,
                                save_folder_path=self.new_folder_path)
    
    def predict(self,
                model_path: str,
                text_data: list[str] | str | None = None,
                scalar_data: list[float] | float | None = None,
                return_proba: bool = False,
                ) -> str | dict[str, str | float | dict[str, float]]:
        """Предсказывает класс объекта и, при необходимости, вероятности классов.

        Args:
            model_path: Путь к сохранённой модели (.pt).
            text_data: Значения текстовых признаков в порядке text_column.
            scalar_data: Значения числовых признаков в порядке data_column.
            return_proba: Если True, кроме метки вернуть вероятности классов.
                По умолчанию False — поведение как в прежних версиях.

        Returns:
            При return_proba=False — метка предсказанного класса.
            При return_proba=True — словарь:
                'label': метка класса с наибольшей вероятностью;
                'prob': вероятность этого класса (уверенность модели);
                'probs': вероятности всех классов {класс: вероятность},
                    их сумма равна 1.
        """
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = checkpoint['model']
        model.eval()
        label_encoder = checkpoint['label_encoder']

        X = _prepare_inference_data(text_data, scalar_data,
                                    checkpoint.get('vectorizer_text'),
                                    checkpoint.get('vectorizer_scalar'))

        with torch.no_grad():

            probs = torch.softmax(model(X), dim=1)[0]          

        pred_idx = int(probs.argmax())
        label = label_encoder.inverse_transform([pred_idx])[0]

        if not return_proba:
            return label

        probs_list = probs.tolist()                          
        return {
            'label': label,
            'prob': probs_list[pred_idx],
            'probs': dict(zip(label_encoder.classes_.tolist(), probs_list)),
        }

    
    def finetune(self,
                pretrained_path,
                train_path,
                val_path=None,
                num_epochs=10,
                hidden_size=None,
                num_layers=None,
                batch_size=128,
                learning_rate=0.0001,
                dropout=None,
                optim_name='Adam',
                crit_name='CELoss',
                device_name='cuda'):
        """
        Дообучение уже обученной модели.
        
        Args:
            pretrained_path (str): путь к .pt файлу с сохранённой моделью
            train_path (str): путь к обучающему датасету
            val_path (str): путь к валидационному датасету (опционально)
            learning_rate (float): рекомендуется в 10-100 раз меньше, чем при обучении с нуля
            hidden_size (int | None): не используется, берётся из чекпоинта
            num_layers (int | None): не используется, берётся из чекпоинта
            dropout (float | None): новый dropout для дообучения; None - оставить как в модели
        """
        print(f"Загрузка модели из: {pretrained_path}")
    
        # Загружаем чекпоинт
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # Восстанавливаем состояние из чекпоинта
        self.model = checkpoint['model']
        hidden_size, num_layers, dropout = _finetune_arch_params(checkpoint, self.model, hidden_size, num_layers, dropout)
        self.vectorizer_text = checkpoint.get('vectorizer_text')
        self.vectorizer_scalar = checkpoint.get('vectorizer_scalar')
        self.label_encoder = checkpoint['label_encoder']
        self.num_classes = checkpoint['num_classes']
        
        # Загружаем данные
        self.train_path = train_path
        self.val_path = val_path
        self.df_train = pd.read_csv(train_path)
        
        # Векторизуем данные 
        if val_path is None:
            # Векторизация признаков существующими векторайзерами
            features = []
            
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features.append(sparse.csr_matrix(feat))
            
            if not features:
                raise ValueError("Нет векторайзеров в чекпоинте!")
            
            X_all = sparse.hstack(features).tocsr() if len(features) > 1 else features[0]
            
            # Кодирование меток 
            y_all = self.label_encoder.transform(self.df_train[self.label_column])
            
            # Разделение на train/val
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_all, y_all, test_size=self.split_ratio, random_state=self.random_state)
        else:
            # С отдельным валидационным набором
            self.df_val = pd.read_csv(val_path)
            
            # Векторизация train
            features_tr = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features_tr.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features_tr.append(sparse.csr_matrix(feat))
            
            self.X_train = sparse.hstack(features_tr).tocsr() if len(features_tr) > 1 else features_tr[0]
            self.y_train = self.label_encoder.transform(self.df_train[self.label_column])
            
            # Векторизация val
            features_val = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_val[col].fillna(''))
                    features_val.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_val[col].fillna(0).values.reshape(-1, 1))
                    features_val.append(sparse.csr_matrix(feat))
            
            self.X_val = sparse.hstack(features_val).tocsr() if len(features_val) > 1 else features_val[0]
            self.y_val = self.label_encoder.transform(self.df_val[self.label_column])
        
        self.input_size = self.X_train.shape[1]
        
        # Проверка совместимости
        saved_input_size = checkpoint.get('input_size')
        if saved_input_size and saved_input_size != self.input_size:
            raise ValueError(
                f"Несовпадение размерности! В модели: {saved_input_size}, "
                f"в данных: {self.input_size}. Используйте те же параметры векторизации."
            )

        # Инициализация метрик
        self.mtx = Metrics()
        
        # Создание папки для результатов
        self.new_folder_path = dirruns(folder_name="finetune_")
        
        # Создание файла для результатов
        headers = ["epoch", "train_loss", "train_acc, %", "val_loss", "val_acc, %",
                "val_precision, %", "val_recall,%", "F1_score"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        # Создание DataLoader
        train_dataset = ClassifierDataset(self.X_train, self.y_train)
        val_dataset = ClassifierDataset(self.X_val, self.y_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор
        criterion = CLASS_CRITERIA.get(crit_name, nn.CrossEntropyLoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        print(f"Дообучение на устройстве: {device}")
        
        # Создание и запуск тренера
        trainer = ClassificationTrainer(model=self.model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        device=device,
                                        train_loader=train_loader,
                                        val_loader=val_loader,
                                        label_encoder=self.label_encoder,
                                        vectorizer_text=self.vectorizer_text,
                                        vectorizer_scalar=self.vectorizer_scalar,
                                        new_folder_path=self.new_folder_path)


        # Запуск обучения
        metrics = trainer.train(num_epochs=num_epochs,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                input_size=self.input_size,
                                num_classes=self.num_classes,
                                train_path=train_path,
                                label_column=self.label_column,
                                dropout=dropout,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                optim_name=optim_name,
                                crit_name=crit_name)
        
        # Визуализация метрик
        self._visualize_metrics(metrics)
        return
    
    def tune(self,
            model_name,
            train_path,
            val_path,
            iterations,
            num_epoch,
            hidden_size_range=[64, 128, 256],
            num_layers_range=[1, 2, 3],
            learning_rate_range=(1e-5, 1e-2),
            dropout_range=(0.0, 0.5),
            batch_size_range=[32, 64, 128],
            optim_name='Adam',
            crit_name='CELoss',
            device_name='cuda',
            seed=None):
        """
        Подбор оптимальных гиперпараметров с использованием Optuna.
        """
        
        warnings.filterwarnings("ignore", message=".*dropout option adds dropout after all but last recurrent layer.*")
        warnings.filterwarnings("ignore", category=UserWarning)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epoch = num_epoch
        self.optim_name = optim_name
        self.crit_name = crit_name
        self.device_name = device_name
        self.tune_folder_path = dirruns("tune_")
        
        headers = ["iteration", "min_val_loss", "hidden_size", "num_layers", 
                "batch_size", "learning_rate", "dropout"]
        self.csv_tune_path = f"{self.tune_folder_path}/result_tune.csv"
        if not os.path.isfile(self.csv_tune_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_tune_path, index=False)
        
        if seed is not None:
            sampler = TPESampler(seed=seed)
            print(f"Используется фиксированный seed={seed}")
        else:
            sampler = TPESampler()
            print(f"Используется случайный seed")
        
        study = optuna.create_study(
            study_name="lancetnic_classification_tune",
            direction="minimize",
            sampler=sampler
        )
        
        pbar = tqdm(total=iterations, desc="Оптимизация гиперпараметров", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", hidden_size_range)
            num_layers = trial.suggest_categorical("num_layers", num_layers_range)
            learning_rate = trial.suggest_float("learning_rate", 
                                            learning_rate_range[0], 
                                            learning_rate_range[1], 
                                            log=True)
            dropout = trial.suggest_float("dropout", 
                                        dropout_range[0], 
                                        dropout_range[1])
            batch_size = trial.suggest_categorical("batch_size", batch_size_range)
            
            try:
                tuning = self.train(model_name=self.model_name,
                                    train_path=self.train_path,
                                    val_path=self.val_path,
                                    num_epochs=self.num_epoch,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    dropout=dropout,
                                    optim_name=self.optim_name,
                                    crit_name=self.crit_name,
                                    device_name=self.device_name,
                                    folder_name="tune_",
                                    visualize=False, 
                                    save_folder_path=self.tune_folder_path)
                
                min_val_loss = min(tuning['val_loss'])
                
                # Сохранение результатов trial
                hyperparams = {"trial": trial.number + 1,
                               "model_params": {"hidden_size": hidden_size, 
                                                "num_layers": num_layers, "dropout": dropout},
                               "train_params": {"num_epochs": self.num_epoch, 
                                                "batch_size": batch_size, 
                                                "learning_rate": learning_rate, 
                                                "optimizer": self.optim_name, 
                                                "criterion": self.crit_name},
                               "result": {"min_val_loss": min_val_loss}}
                
                trials_dir = f"{self.tune_folder_path}/trials"
                os.makedirs(trials_dir, exist_ok=True)

                path_config = f"{trials_dir}/tune_hyperparams_trial_{trial.number + 1}.yaml"
                with open(path_config, 'w', encoding='utf-8') as f:
                    yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
                
                row = {"iteration": trial.number + 1,
                       "min_val_loss": f"{min_val_loss:.4f}",
                       "hidden_size": hidden_size,
                       "num_layers": num_layers,
                       "batch_size": batch_size,
                       "learning_rate": f"{learning_rate:.2e}",
                       "dropout": f"{dropout:.3f}"}
                
                pd.DataFrame([row], columns=headers).to_csv(self.csv_tune_path, mode='a', header=False, index=False)
                
                return min_val_loss
                
            except Exception as e:
                print(f"\n✗ Ошибка в trial {trial.number + 1}: {e}")
                return float('inf')
        
        def progress_callback(study, trial):
            pbar.update(1)
            if study.best_trial:
                pbar.set_postfix({'best_loss': f'{study.best_trial.value:.4f}'})

        try:
            study.optimize(objective, n_trials=iterations, show_progress_bar=False, callbacks=[progress_callback])
        except KeyboardInterrupt:
            print("\n⚠️ Оптимизация прервана пользователем.")
        finally:
            pbar.close()
        
        if not study.trials:
            print("Не удалось завершить ни один trial.")
            return None

        best_trial = study.best_trial
        
        print("\n" + "="*70)
        print("🏆 ПОДБОР ЗАВЕРШЕН!")
        print("="*70)
        print(f"Лучший trial: #{best_trial.number + 1}")
        print(f"Лучший min_val_loss: {best_trial.value:.4f}")
        print("Лучшие параметры:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("="*70)
        
        best_params = best_trial.params.copy()
        best_params['num_epochs'] = self.num_epoch
        best_params['optimizer'] = self.optim_name
        best_params['criterion'] = self.crit_name
        
        best_config = {"best_trial_number": best_trial.number + 1,
                       "best_val_loss": best_trial.value,
                       "best_parameters": best_params,
                       "seed_used": seed}
        
        best_path = f"{self.tune_folder_path}/best_hyperparams.yaml"
        with open(best_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, indent=2)
        
        print(f"\n📁 Лучшие параметры: {best_path}")
        print(f"📊 Все trials: {self.csv_tune_path}")
        
        return best_params
            
class MultiLabel(BaseClassificationPipeline):
    """Мультилейбл классификация (несколько классов на объект)."""
    
    _dataset_class = MultiLabelDataset
    _trainer_class = MultiLabelTrainer
    _criteria_dict = MULTI_LABEL_CRITERIA
    _default_criterion = 'BCEWithLogitsLoss'
    
    def __init__(self, 
                 text_column=None, 
                 data_column=None, 
                 label_column=None,
                 label_separator=';', 
                 split_ratio=0.2, 
                 random_state=42, 
                 max_features=None,
                 text_mode='tfidf',
                 max_len=100):
        
        super().__init__(text_column, 
                         data_column, 
                         label_column,
                         split_ratio, 
                         random_state, 
                         max_features,
                         text_mode=text_mode,
                         max_len=max_len)
        
        self.label_separator = label_separator
        self.label_binarizer = None
        self.threshold = 0.5
    
    def _parse_labels(self, series):
        """Парсинг мультилейбл меток (строка с разделителем -> список)."""
        result = []
        for val in series:
            if pd.isna(val):
                result.append([])
            elif isinstance(val, list):
                result.append(val)
            elif isinstance(val, str):
                parts = [p.strip() for p in val.split(self.label_separator) if p.strip()]
                result.append(parts)
            else:
                result.append([str(val)])
        return result
    
    def _encode_labels(self) -> None:
        """Кодирует наборы меток в multi-hot векторы"""
        self.label_binarizer = MultiLabelBinarizer()
        y_tr_lists = self._parse_labels(self.df_train[self.label_column])
        y_val_lists = self._parse_labels(self.df_val[self.label_column])
        self.label_binarizer.fit(y_tr_lists + y_val_lists)
        self.y_train = self.label_binarizer.transform(y_tr_lists)
        self.y_val = self.label_binarizer.transform(y_val_lists)
        self.num_classes = len(self.label_binarizer.classes_)
    
    def _get_trainer_kwargs(self) -> dict:
        return {'label_binarizer': self.label_binarizer,
                'threshold': self.threshold}
    
    def _get_csv_headers(self):
        return ["epoch", "train_loss", "val_loss",
                "train_subset_acc, %", "val_subset_acc, %",
                "val_hamming_loss", "val_precision, %", "val_recall, %", "val_f1, %"]
    
    def _visualize_metrics(self, metrics):
        self.mtx.multi_label_confusion_matrix(last_labels=metrics['all_labels'][-1], 
                                              last_probs=metrics['all_probs'][-1],
                                              label_binarizer=self.label_binarizer,
                                              save_folder_path=self.new_folder_path,
                                              plt_name="confusion_matrix_last_model", 
                                              threshold=self.threshold)
        
        self.mtx.train_val_loss(epoch=metrics['epoch'], 
                                train_loss=metrics['train_loss'],
                                val_loss=metrics['val_loss'], 
                                save_folder_path=self.new_folder_path)
        
        self.mtx.multi_label_subset_acc(epoch=metrics['epoch'],
                                        train_acc=metrics['train_subset_acc'],
                                        val_acc=metrics['val_subset_acc'],
                                        save_folder_path=self.new_folder_path)
        
        self.mtx.hamming_loss_plot(epoch=metrics['epoch'],
                                   hamming_loss=metrics['val_hamming_loss'],
                                   save_folder_path=self.new_folder_path)
        
        self.mtx.multi_label_prf_plot(epoch=metrics['epoch'],
                                      precision=metrics['val_precision'],
                                      recall=metrics['val_recall'],
                                      f1=metrics['val_f1'],
                                      save_folder_path=self.new_folder_path)
        
        self.mtx.per_class_f1(last_labels=metrics['all_labels'][-1],
                              last_preds=metrics['all_preds'][-1],
                              label_binarizer=self.label_binarizer,
                              save_folder_path=self.new_folder_path)
        
        self.mtx.label_distribution(data_path=self.train_path,
                                    label_column=self.label_column,
                                    label_separator=self.label_separator,
                                    save_folder_path=self.new_folder_path)
    
    def train(self, model_name, train_path, val_path, num_epochs, **kwargs):
        self.threshold = kwargs.pop('threshold', 0.5)
        return super().train(model_name, train_path, val_path, num_epochs, **kwargs)
    
    def predict(self, model_path, text_data=None, scalar_data=None, threshold=None, top_k=None):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = checkpoint['model']
        model.eval()
        label_binarizer = checkpoint['label_binarizer']
        thr = threshold if threshold is not None else checkpoint.get('threshold', 0.5)
        
        X = _prepare_inference_data(
            text_data, scalar_data,
            checkpoint.get('vectorizer_text'), checkpoint.get('vectorizer_scalar'))
        
        with torch.no_grad():
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        mask = probs >= thr
        classes = label_binarizer.classes_
        
        if top_k is not None:
            idx = np.argsort(probs)[::-1][:top_k]
            mask = np.zeros_like(probs, dtype=bool)
            mask[idx] = True
        
        pred_labels = classes[mask].tolist()
        pred_probs = {cls: float(probs[i]) for i, cls in enumerate(classes) if mask[i]}
        return {'labels': pred_labels, 'probs': pred_probs}

    def finetune(self,
                pretrained_path,
                train_path,
                val_path=None,
                num_epochs=10,
                hidden_size=None,
                num_layers=None,
                batch_size=128,
                learning_rate=0.0001,
                dropout=None,
                optim_name='Adam',
                crit_name='BCEWithLogitsLoss',
                device_name='cuda',
                threshold=None):
        """Дообучение мультилейбл модели."""
        print(f"Загрузка модели из: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        self.model = checkpoint['model']
        hidden_size, num_layers, dropout = _finetune_arch_params(checkpoint, self.model, hidden_size, num_layers, dropout)
        self.vectorizer_text = checkpoint.get('vectorizer_text')
        self.vectorizer_scalar = checkpoint.get('vectorizer_scalar')
        self.label_binarizer = checkpoint['label_binarizer']
        self.num_classes = checkpoint['num_classes']
        self.threshold = threshold if threshold is not None else checkpoint.get('threshold', 0.5)
        
        self.train_path = train_path
        self.val_path = val_path
        self.df_train = pd.read_csv(train_path)
        
        # Векторизация
        if val_path is None:
            features = []
            
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features.append(sparse.csr_matrix(feat))
            
            X_all = sparse.hstack(features).tocsr() if len(features) > 1 else features[0]
            
            y_all_lists = self._parse_labels(self.df_train[self.label_column])
            y_all = self.label_binarizer.transform(y_all_lists)
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_all, y_all, test_size=self.split_ratio, random_state=self.random_state)
        else:
            self.df_val = pd.read_csv(val_path)
            
            # Векторизация train
            features_tr = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features_tr.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features_tr.append(sparse.csr_matrix(feat))
            
            self.X_train = sparse.hstack(features_tr).tocsr() if len(features_tr) > 1 else features_tr[0]
            y_train_lists = self._parse_labels(self.df_train[self.label_column])
            self.y_train = self.label_binarizer.transform(y_train_lists)
            
            # Векторизация val
            features_val = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_val[col].fillna(''))
                    features_val.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_val[col].fillna(0).values.reshape(-1, 1))
                    features_val.append(sparse.csr_matrix(feat))
            
            self.X_val = sparse.hstack(features_val).tocsr() if len(features_val) > 1 else features_val[0]
            y_val_lists = self._parse_labels(self.df_val[self.label_column])
            self.y_val = self.label_binarizer.transform(y_val_lists)
        
        self.input_size = self.X_train.shape[1]
        
        saved_input_size = checkpoint.get('input_size')
        if saved_input_size and saved_input_size != self.input_size:
            raise ValueError(f"Несовпадение размерности! В модели: {saved_input_size}, в данных: {self.input_size}")
        
        self.mtx = Metrics()
        self.new_folder_path = dirruns(folder_name="finetune_")
        
        headers = ["epoch", "train_loss", "val_loss", "train_subset_acc, %", "val_subset_acc, %",
                "val_hamming_loss", "val_precision, %", "val_recall, %", "val_f1, %"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        # DataLoader
        train_dataset = MultiLabelDataset(self.X_train, self.y_train)
        val_dataset = MultiLabelDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор
        criterion = MULTI_LABEL_CRITERIA.get(crit_name, nn.BCEWithLogitsLoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        print(f"Дообучение на устройстве: {device}")
        
        # Создание тренера
        trainer = MultiLabelTrainer(model=self.model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    device=device,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    label_binarizer=self.label_binarizer,
                                    vectorizer_text=self.vectorizer_text,
                                    vectorizer_scalar=self.vectorizer_scalar,
                                    new_folder_path=self.new_folder_path,
                                    threshold=self.threshold)
        
        metrics = trainer.train(num_epochs=num_epochs,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                input_size=self.input_size,
                                num_classes=self.num_classes,
                                train_path=train_path,
                                label_column=self.label_column,
                                dropout=dropout,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                optim_name=optim_name,
                                crit_name=crit_name,
                                threshold=self.threshold)
        
        self._visualize_metrics(metrics)
        return 

    def tune(self,
            model_name,
            train_path,
            val_path,
            iterations,
            num_epoch,
            hidden_size_range=[64, 128, 256],
            num_layers_range=[1, 2, 3],
            learning_rate_range=(1e-5, 1e-2),
            dropout_range=(0.0, 0.5),
            batch_size_range=[32, 64, 128],
            optim_name='Adam',
            crit_cls_name='CELoss',         
            crit_reg_name='MSELoss',         
            device_name='cuda',
            loss_ratio_cls=1.0,              
            loss_ratio_reg=1.0,              
            seed=None):
        """
        Подбор оптимальных гиперпараметров с использованием Optuna.
        """
        
        warnings.filterwarnings("ignore", message=".*dropout option adds dropout after all but last recurrent layer.*")
        warnings.filterwarnings("ignore", category=UserWarning)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epoch = num_epoch
        self.optim_name = optim_name
        self.crit_cls_name = crit_cls_name        
        self.crit_reg_name = crit_reg_name        
        self.device_name = device_name
        self.loss_ratio_cls = loss_ratio_cls      
        self.loss_ratio_reg = loss_ratio_reg      
        self.tune_folder_path = dirruns("tune_")
        
        headers = ["iteration", "min_val_loss", "hidden_size", "num_layers", 
                "batch_size", "learning_rate", "dropout"]
        self.csv_tune_path = f"{self.tune_folder_path}/result_tune.csv"
        if not os.path.isfile(self.csv_tune_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_tune_path, index=False)
        
        if seed is not None:
            sampler = TPESampler(seed=seed)
            print(f"Используется фиксированный seed={seed}")
        else:
            sampler = TPESampler()
            print(f"Используется случайный seed")
        
        study = optuna.create_study(
            study_name="lancetnic_multitask_tune",   
            direction="minimize",
            sampler=sampler
        )
        
        pbar = tqdm(total=iterations, desc="Оптимизация гиперпараметров", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", hidden_size_range)
            num_layers = trial.suggest_categorical("num_layers", num_layers_range)
            learning_rate = trial.suggest_float("learning_rate", 
                                            learning_rate_range[0], 
                                            learning_rate_range[1], 
                                            log=True)
            dropout = trial.suggest_float("dropout", 
                                        dropout_range[0], 
                                        dropout_range[1])
            batch_size = trial.suggest_categorical("batch_size", batch_size_range)
            
            try:
                tuning = self.train(model_name=self.model_name,
                                    train_path=self.train_path,
                                    val_path=self.val_path,
                                    num_epochs=self.num_epoch,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    dropout=dropout,
                                    optim_name=self.optim_name,
                                    crit_cls_name=self.crit_cls_name,     
                                    crit_reg_name=self.crit_reg_name,     
                                    device_name=self.device_name,
                                    loss_ratio_cls=self.loss_ratio_cls,   
                                    loss_ratio_reg=self.loss_ratio_reg,   
                                    folder_name="tune_",
                                    visualize=False,
                                    save_folder_path=self.tune_folder_path)
                
                min_val_loss = min(tuning['val_loss'])
                
                # Сохранение результатов trial
                hyperparams = {"trial": trial.number + 1,
                               "model_params": {"hidden_size": hidden_size, 
                                                "num_layers": num_layers, "dropout": dropout},
                               "train_params": {"num_epochs": self.num_epoch, 
                                                "batch_size": batch_size, 
                                                "learning_rate": learning_rate, 
                                                "optimizer": self.optim_name, 
                                                "criterion_cls": self.crit_cls_name,    
                                                "criterion_reg": self.crit_reg_name,    
                                                "loss_ratio_cls": self.loss_ratio_cls,  
                                                "loss_ratio_reg": self.loss_ratio_reg}, 
                               "result": {"min_val_loss": min_val_loss}}
                
                trials_dir = f"{self.tune_folder_path}/trials"
                os.makedirs(trials_dir, exist_ok=True)

                path_config = f"{trials_dir}/tune_hyperparams_trial_{trial.number + 1}.yaml"
                with open(path_config, 'w', encoding='utf-8') as f:
                    yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
                
                row = {"iteration": trial.number + 1,
                       "min_val_loss": f"{min_val_loss:.4f}",
                       "hidden_size": hidden_size,
                       "num_layers": num_layers,
                       "batch_size": batch_size,
                       "learning_rate": f"{learning_rate:.2e}",
                       "dropout": f"{dropout:.3f}"}
                
                pd.DataFrame([row], columns=headers).to_csv(self.csv_tune_path, mode='a', header=False, index=False)
                
                return min_val_loss
                
            except Exception as e:
                print(f"\n✗ Ошибка в trial {trial.number + 1}: {e}")
                return float('inf')
        
        def progress_callback(study, trial):
            pbar.update(1)
            if study.best_trial:
                pbar.set_postfix({'best_loss': f'{study.best_trial.value:.4f}'})

        try:
            study.optimize(objective, n_trials=iterations, show_progress_bar=False, callbacks=[progress_callback])
        except KeyboardInterrupt:
            print("\n⚠️ Оптимизация прервана пользователем.")
        finally:
            pbar.close()
        
        if not study.trials:
            print("Не удалось завершить ни один trial.")
            return None

        best_trial = study.best_trial
        
        print("\n" + "="*70)
        print("🏆 ПОДБОР ЗАВЕРШЕН!")
        print("="*70)
        print(f"Лучший trial: #{best_trial.number + 1}")
        print(f"Лучший min_val_loss: {best_trial.value:.4f}")
        print("Лучшие параметры:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("="*70)
        
        best_params = best_trial.params.copy()
        best_params['num_epochs'] = self.num_epoch
        best_params['optimizer'] = self.optim_name
        best_params['criterion_cls'] = self.crit_cls_name      
        best_params['criterion_reg'] = self.crit_reg_name      
        best_params['loss_ratio_cls'] = self.loss_ratio_cls    
        best_params['loss_ratio_reg'] = self.loss_ratio_reg    
        
        best_config = {"best_trial_number": best_trial.number + 1,
                       "best_val_loss": best_trial.value,
                       "best_parameters": best_params,
                       "seed_used": seed}
        
        best_path = f"{self.tune_folder_path}/best_hyperparams.yaml"
        with open(best_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, indent=2)
        
        print(f"\n📁 Лучшие параметры: {best_path}")
        print(f"📊 Все trials: {self.csv_tune_path}")
        
        return best_params
    
# =====================================================================
# РЕГРЕССИЯ 
# =====================================================================

class Regression:
    def __init__(self, 
                 text_column=None, 
                 data_column=None, 
                 label_column=None,
                 split_ratio=0.2, 
                 random_state=42, 
                 max_features=None,
                 text_mode='tfidf',
                 max_len=100):
        
        self.text_column = text_column
        self.data_column = data_column
        self.label_column = label_column
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.max_features = max_features
        self.text_mode = text_mode
        self.max_len = max_len
        self.df_train = None
        self.df_val = None
        self.vectorizer_text = None
        self.vectorizer_scalar = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.input_size = None
        self.output_size = 1
        self.scaler_reg = None
        self.model = None
        self.new_folder_path = None
        self.mtx = Metrics()

    def _scale_target(self, values: np.ndarray) -> np.ndarray:
        """Нормирует целевую переменную уже обученным scaler_reg"""
        if self.scaler_reg is None:
            return values
        return self.scaler_reg.transform(values.reshape(-1, 1)).ravel()
    
    def vectorize_with_val_path(self) -> None:
        self.df_val = pd.read_csv(self.val_path)
        X_tr, X_val, vec_t, vec_s = _extract_features(
            self.text_column, self.data_column,
            self.df_train, self.df_val, self.max_features,
            self.text_mode, self.max_len)
        self.X_train, self.X_val = X_tr, X_val
        self.vectorizer_text, self.vectorizer_scalar = vec_t, vec_s
        self.scaler_reg = StandardScaler()
        self.scaler_reg.fit(self.df_train[self.label_column].values.reshape(-1, 1))
        self.y_train = self._scale_target(self.df_train[self.label_column].values)
        self.y_val = self._scale_target(self.df_val[self.label_column].values)
        self.input_size = self.X_train.shape[1]
    
    def vectorize_no_val_path(self) -> None:
        """Векторизация без отдельного валидационного набора"""
        self.df_train, self.df_val = train_test_split(
            self.df_train, test_size=self.split_ratio, random_state=self.random_state)
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_val = self.df_val.reset_index(drop=True)

        X_tr, X_val, vec_t, vec_s = _extract_features(
            self.text_column, self.data_column,
            self.df_train, self.df_val, self.max_features,
            self.text_mode, self.max_len)
        self.X_train, self.X_val = X_tr, X_val
        self.vectorizer_text, self.vectorizer_scalar = vec_t, vec_s
        self.scaler_reg = StandardScaler()
        self.scaler_reg.fit(self.df_train[self.label_column].values.reshape(-1, 1))
        self.y_train = self._scale_target(self.df_train[self.label_column].values)
        self.y_val = self._scale_target(self.df_val[self.label_column].values)
        self.input_size = self.X_train.shape[1]
    
    def train(self, 
              model_name, 
              train_path, 
              val_path, 
              num_epochs,
              hidden_size=256, 
              num_layers=1, 
              batch_size=128, 
              learning_rate=0.001,
              dropout=0, 
              optim_name='Adam', 
              crit_name='MSELoss', 
              device_name='cuda',
              folder_name='train_',
              visualize=True,            
              save_folder_path=None):
        
        self.train_path = train_path
        self.val_path = val_path
        self.df_train = pd.read_csv(train_path)
        
        if val_path is None:
            self.vectorize_no_val_path()
        else:
            self.vectorize_with_val_path()
        
        if save_folder_path is None:
            self.new_folder_path = dirruns(folder_name)
        else:
            self.new_folder_path = save_folder_path

        headers = ["epoch", "train_loss", "val_loss", "train_mae", "val_mae", "train_rmse", "val_rmse"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if visualize and not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        train_dataset = RegressionDataset(self.X_train, self.y_train)
        val_dataset = RegressionDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.model = model_name(self.input_size, hidden_size, num_layers, self.output_size, dropout,
                                **_sequence_model_kwargs(self.text_mode, self.vectorizer_text, self.max_len))
        criterion = REG_CRITERIA.get(crit_name, nn.MSELoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        
        trainer = RegressionTrainer(model=self.model, 
                                    criterion=criterion, 
                                    optimizer=optimizer, 
                                    device=device,
                                    train_loader=train_loader, 
                                    val_loader=val_loader,
                                    vectorizer_text=self.vectorizer_text, 
                                    vectorizer_scalar=self.vectorizer_scalar,
                                    new_folder_path=self.new_folder_path,
                                    scaler_reg=self.scaler_reg)
        
        trainer.visualize = visualize
        
        metrics = trainer.train(num_epochs=num_epochs, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers,
                                input_size=self.input_size, 
                                output_size=self.output_size,
                                train_path=train_path, 
                                label_column=self.label_column,
                                dropout=dropout, 
                                batch_size=batch_size, 
                                learning_rate=learning_rate,
                                optim_name=optim_name, 
                                crit_name=crit_name)
        
        if visualize:
            self.visualize_metrics(metrics)
            return None
        return metrics
    
    def visualize_metrics(self, metrics):
        self.mtx.train_val_loss(epoch=metrics['epoch'], 
                                train_loss=metrics['train_loss'],
                                val_loss=metrics['val_loss'], 
                                save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_mae(epoch=metrics['epoch'], 
                               train_mae=metrics['train_mae'],
                               val_mae=metrics['val_mae'], 
                               save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_rmse(epoch=metrics['epoch'], 
                                train_rmse=metrics['train_rmse'],
                                val_rmse=metrics['val_rmse'], 
                                save_folder_path=self.new_folder_path)
        
        self.mtx.regression_scatter(metrics=metrics, 
                                    save_folder_path=self.new_folder_path)
    
    def predict(self, model_path, text_data=None, scalar_data=None):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = checkpoint['model']
        model.eval()
        X = _prepare_inference_data(text_data, scalar_data, checkpoint.get('vectorizer_text'), checkpoint.get('vectorizer_scalar'))
        scaler_reg = checkpoint.get('scaler_reg')
        with torch.no_grad():
            pred = model(X).item()
        if scaler_reg is not None:
            pred = float(scaler_reg.inverse_transform([[pred]])[0, 0])
        return pred

    def finetune(self,
                pretrained_path,
                train_path,
                val_path=None,
                num_epochs=10,
                hidden_size=None,
                num_layers=None,
                batch_size=128,
                learning_rate=0.0001,
                dropout=None,
                optim_name='Adam',
                crit_name='MSELoss',
                device_name='cuda'):
        """Дообучение модели регрессии."""
        print(f"Загрузка модели из: {pretrained_path}")
        
        # Загрузка чекпоинта
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        self.model = checkpoint['model']
        hidden_size, num_layers, dropout = _finetune_arch_params(checkpoint, self.model, hidden_size, num_layers, dropout)
        self.vectorizer_text = checkpoint.get('vectorizer_text')
        self.vectorizer_scalar = checkpoint.get('vectorizer_scalar')
        self.scaler_reg = checkpoint.get('scaler_reg')
        self.output_size = checkpoint.get('output_size', 1)
        
        self.train_path = train_path
        self.val_path = val_path
        self.df_train = pd.read_csv(train_path)
        
        # Векторизация
        if val_path is None:
            features = []
            
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features.append(sparse.csr_matrix(feat))
            
            X_all = sparse.hstack(features).tocsr() if len(features) > 1 else features[0]
            y_all = self._scale_target(self.df_train[self.label_column].values)
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_all, y_all, test_size=self.split_ratio, random_state=self.random_state)
        else:
            self.df_val = pd.read_csv(val_path)
            
            # Векторизация train
            features_tr = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features_tr.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features_tr.append(sparse.csr_matrix(feat))
            
            self.X_train = sparse.hstack(features_tr).tocsr() if len(features_tr) > 1 else features_tr[0]
            self.y_train = self._scale_target(self.df_train[self.label_column].values)
            
            # Векторизация val
            features_val = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_val[col].fillna(''))
                    features_val.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_val[col].fillna(0).values.reshape(-1, 1))
                    features_val.append(sparse.csr_matrix(feat))
            
            self.X_val = sparse.hstack(features_val).tocsr() if len(features_val) > 1 else features_val[0]
            self.y_val = self._scale_target(self.df_val[self.label_column].values)
        
        self.input_size = self.X_train.shape[1]
        
        saved_input_size = checkpoint.get('input_size')
        if saved_input_size and saved_input_size != self.input_size:
            raise ValueError(f"Несовпадение размерности! В модели: {saved_input_size}, в данных: {self.input_size}")
        
        self.mtx = Metrics()
        self.new_folder_path = dirruns(folder_name="finetune_")
        
        headers = ["epoch", "val_loss", "train_loss", "train_mae", "val_mae", "train_rmse", "val_rmse"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        # DataLoader
        train_dataset = RegressionDataset(self.X_train, self.y_train)
        val_dataset = RegressionDataset(self.X_val, self.y_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор
        criterion = REG_CRITERIA.get(crit_name, nn.MSELoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        print(f"Дообучение на устройстве: {device}")
        
        # Тренер
        trainer = RegressionTrainer(model=self.model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    device=device,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    vectorizer_text=self.vectorizer_text,
                                    vectorizer_scalar=self.vectorizer_scalar,
                                    new_folder_path=self.new_folder_path)
        
        metrics = trainer.train(num_epochs=num_epochs,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                input_size=self.input_size,
                                output_size=self.output_size,
                                train_path=train_path,
                                label_column=self.label_column,
                                dropout=dropout,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                optim_name=optim_name,
                                crit_name=crit_name)
        
        self.visualize_metrics(metrics)
        return

    def tune(self,
            model_name,
            train_path,
            val_path,
            iterations,
            num_epoch,
            hidden_size_range=[64, 128, 256],
            num_layers_range=[1, 2, 3],
            learning_rate_range=(1e-5, 1e-2),
            dropout_range=(0.0, 0.5),
            batch_size_range=[32, 64, 128],
            optim_name='Adam',
            crit_name='MSELoss',
            device_name='cuda',
            seed=None):
        """
        Подбор оптимальных гиперпараметров с использованием Optuna.
        """
        
        warnings.filterwarnings("ignore", message=".*dropout option adds dropout after all but last recurrent layer.*")
        warnings.filterwarnings("ignore", category=UserWarning)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epoch = num_epoch
        self.optim_name = optim_name
        self.crit_name = crit_name
        self.device_name = device_name
        self.tune_folder_path = dirruns("tune_")
        
        headers = ["iteration", "min_val_loss", "hidden_size", "num_layers", 
                "batch_size", "learning_rate", "dropout"]
        self.csv_tune_path = f"{self.tune_folder_path}/result_tune.csv"
        if not os.path.isfile(self.csv_tune_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_tune_path, index=False)
        
        if seed is not None:
            sampler = TPESampler(seed=seed)
            print(f"Используется фиксированный seed={seed}")
        else:
            sampler = TPESampler()
            print(f"Используется случайный seed")
        
        study = optuna.create_study(
            study_name="lancetnic_regression_tune",
            direction="minimize",
            sampler=sampler
        )
        
        pbar = tqdm(total=iterations, desc="Оптимизация гиперпараметров", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", hidden_size_range)
            num_layers = trial.suggest_categorical("num_layers", num_layers_range)
            learning_rate = trial.suggest_float("learning_rate", 
                                            learning_rate_range[0], 
                                            learning_rate_range[1], 
                                            log=True)
            dropout = trial.suggest_float("dropout", 
                                        dropout_range[0], 
                                        dropout_range[1])
            batch_size = trial.suggest_categorical("batch_size", batch_size_range)
            
            try:
                tuning = self.train(model_name=self.model_name,
                                    train_path=self.train_path,
                                    val_path=self.val_path,
                                    num_epochs=self.num_epoch,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    dropout=dropout,
                                    optim_name=self.optim_name,
                                    crit_name=self.crit_name,
                                    device_name=self.device_name,
                                    folder_name="tune_",
                                    visualize=False,
                                    save_folder_path=self.tune_folder_path)
                
                min_val_loss = min(tuning['val_loss'])
                
                # Сохранение результатов trial
                hyperparams = {"trial": trial.number + 1,
                               "model_params": {"hidden_size": hidden_size, 
                                                "num_layers": num_layers, "dropout": dropout},
                               "train_params": {"num_epochs": self.num_epoch, 
                                                "batch_size": batch_size, 
                                                "learning_rate": learning_rate, 
                                                "optimizer": self.optim_name, 
                                                "criterion": self.crit_name},
                               "result": {"min_val_loss": min_val_loss}}
                
                trials_dir = f"{self.tune_folder_path}/trials"
                os.makedirs(trials_dir, exist_ok=True)

                path_config = f"{trials_dir}/tune_hyperparams_trial_{trial.number + 1}.yaml"
                with open(path_config, 'w', encoding='utf-8') as f:
                    yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
                
                row = {"iteration": trial.number + 1,
                       "min_val_loss": f"{min_val_loss:.4f}",
                       "hidden_size": hidden_size,
                       "num_layers": num_layers,
                       "batch_size": batch_size,
                       "learning_rate": f"{learning_rate:.2e}",
                       "dropout": f"{dropout:.3f}"}
                
                pd.DataFrame([row], columns=headers).to_csv(self.csv_tune_path, mode='a', header=False, index=False)
                
                return min_val_loss
                
            except Exception as e:
                print(f"\n✗ Ошибка в trial {trial.number + 1}: {e}")
                return float('inf')
        
        def progress_callback(study, trial):
            pbar.update(1)
            if study.best_trial:
                pbar.set_postfix({'best_loss': f'{study.best_trial.value:.4f}'})

        try:
            study.optimize(objective, n_trials=iterations, show_progress_bar=False, callbacks=[progress_callback])
        except KeyboardInterrupt:
            print("\n⚠️ Оптимизация прервана пользователем.")
        finally:
            pbar.close()
        
        if not study.trials:
            print("Не удалось завершить ни один trial.")
            return None

        best_trial = study.best_trial
        
        print("\n" + "="*70)
        print("🏆 ПОДБОР ЗАВЕРШЕН!")
        print("="*70)
        print(f"Лучший trial: #{best_trial.number + 1}")
        print(f"Лучший min_val_loss: {best_trial.value:.4f}")
        print("Лучшие параметры:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("="*70)
        
        best_params = best_trial.params.copy()
        best_params['num_epochs'] = self.num_epoch
        best_params['optimizer'] = self.optim_name
        best_params['criterion'] = self.crit_name
        
        best_config = {"best_trial_number": best_trial.number + 1,
                       "best_val_loss": best_trial.value,
                       "best_parameters": best_params,
                       "seed_used": seed}
        
        best_path = f"{self.tune_folder_path}/best_hyperparams.yaml"
        with open(best_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, indent=2)
        
        print(f"\n📁 Лучшие параметры: {best_path}")
        print(f"📊 Все trials: {self.csv_tune_path}")
        
        return best_params
    
# =====================================================================
# МУЛЬТИЗАДАЧНАЯ МОДЕЛЬ (КЛАССИФИКАЦИЯ + РЕГРЕССИЯ)
# =====================================================================

class MultiTask:
    def __init__(self, 
                 text_column=None, 
                 data_column=None,
                 label_class=None, 
                 label_reg=None,
                 split_ratio=0.2, 
                 random_state=42, 
                 max_features=None,
                 text_mode='tfidf',
                 max_len=100):
        
        self.text_column = text_column
        self.data_column = data_column
        self.label_class = label_class
        self.label_reg = label_reg
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.max_features = max_features
        self.text_mode = text_mode
        self.max_len = max_len
        self.df_train = None
        self.df_val = None
        self.vectorizer_text = None
        self.vectorizer_scalar = None
        self.label_encoder = None
        self.scaler_reg = None
        self.X_train = None
        self.X_val = None
        self.y_class_train = None
        self.y_class_val = None
        self.y_reg_train = None
        self.y_reg_val = None
        self.input_size = None
        self.num_classes = None
        self.new_folder_path = None
        self.mtx = Metrics()
    
    def _col(self, col):
        return col[0] if isinstance(col, list) else col
    
    def vectorize_with_val_path(self) -> None:
        self.df_val = pd.read_csv(self.val_path)
        X_tr, X_val, vec_t, vec_s = _extract_features(
            self.text_column, self.data_column,
            self.df_train, self.df_val, self.max_features,
            self.text_mode, self.max_len)
        self.X_train, self.X_val = X_tr, X_val
        self.vectorizer_text, self.vectorizer_scalar = vec_t, vec_s
        
        cls_col = self._col(self.label_class)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(pd.concat([self.df_train[cls_col], self.df_val[cls_col]]))
        self.y_class_train = self.label_encoder.transform(self.df_train[cls_col])
        self.y_class_val = self.label_encoder.transform(self.df_val[cls_col])
        
        self.scaler_reg = StandardScaler()
        self.y_reg_train = self.scaler_reg.fit_transform(
            self.df_train[self._col(self.label_reg)].values.reshape(-1, 1)).flatten()
        self.y_reg_val = self.scaler_reg.transform(
            self.df_val[self._col(self.label_reg)].values.reshape(-1, 1)).flatten()
        
        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
    
    def vectorize_no_val_path(self) -> None:
        """Векторизация без отдельного валидационного набора"""
        self.df_train, self.df_val = train_test_split(
            self.df_train, test_size=self.split_ratio, random_state=self.random_state)
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_val = self.df_val.reset_index(drop=True)

        X_tr, X_val, vec_t, vec_s = _extract_features(
            self.text_column, self.data_column,
            self.df_train, self.df_val, self.max_features,
            self.text_mode, self.max_len)
        self.X_train, self.X_val = X_tr, X_val
        self.vectorizer_text, self.vectorizer_scalar = vec_t, vec_s

        cls_col = self._col(self.label_class)
        reg_col = self._col(self.label_reg)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(pd.concat([self.df_train[cls_col], self.df_val[cls_col]]))
        self.y_class_train = self.label_encoder.transform(self.df_train[cls_col])
        self.y_class_val = self.label_encoder.transform(self.df_val[cls_col])

        self.scaler_reg = StandardScaler()
        self.y_reg_train = self.scaler_reg.fit_transform(
            self.df_train[reg_col].values.reshape(-1, 1)).flatten()
        self.y_reg_val = self.scaler_reg.transform(
            self.df_val[reg_col].values.reshape(-1, 1)).flatten()

        self.input_size = self.X_train.shape[1]
        self.num_classes = len(self.label_encoder.classes_)
    
    def train(self, 
              model_name, 
              train_path, 
              val_path, 
              num_epochs,
              hidden_size=256, 
              num_layers=1, 
              batch_size=128, 
              learning_rate=0.001,
              dropout=0, 
              optim_name='Adam', 
              crit_cls_name='CELoss',
              crit_reg_name='MSELoss', 
              device_name='cuda',
              loss_ratio_cls=1.0, 
              loss_ratio_reg=1.0,
              folder_name='train_',
              visualize=True,
              save_folder_path=None):
              
        self.df_train = pd.read_csv(train_path)
        
        if val_path is None:
            self.vectorize_no_val_path()
        else:
            self.vectorize_with_val_path()
        
        if save_folder_path is None:
            self.new_folder_path = dirruns(folder_name)
        else:
            self.new_folder_path = save_folder_path

        headers = ["epoch", "train_loss", "val_loss", "val_acc", "f1_score", "val_mae", "val_rmse"]
        csv_path = f"{self.new_folder_path}/result.csv"
        if visualize and not os.path.isfile(csv_path):
            pd.DataFrame(columns=headers).to_csv(csv_path, index=False)
        
        train_dataset = MultiTaskDataset(self.X_train, self.y_class_train, self.y_reg_train)
        val_dataset = MultiTaskDataset(self.X_val, self.y_class_val, self.y_reg_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.model = model_name(self.input_size, hidden_size, num_layers, self.num_classes, dropout,
                                **_sequence_model_kwargs(self.text_mode, self.vectorizer_text, self.max_len))
        crit_class = CLASS_CRITERIA.get(crit_cls_name, nn.CrossEntropyLoss)()
        crit_reg = REG_CRITERIA.get(crit_reg_name, nn.MSELoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        
        trainer = MultiTaskTrainer(model=self.model, 
                                   criterion_class=crit_class, 
                                   criterion_reg=crit_reg,
                                   optimizer=optimizer, 
                                   device=device,
                                   train_loader=train_loader, 
                                   val_loader=val_loader,
                                   label_encoder=self.label_encoder,
                                   vectorizer_text=self.vectorizer_text, 
                                   vectorizer_scalar=self.vectorizer_scalar,
                                   new_folder_path=self.new_folder_path,
                                   loss_ratio_cls=loss_ratio_cls, 
                                   loss_ratio_reg=loss_ratio_reg,
                                   scaler_reg=self.scaler_reg)
        
        trainer.visualize = visualize
        
        metrics = trainer.train(num_epochs=num_epochs, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers,
                                input_size=self.input_size, 
                                num_classes=self.num_classes,
                                train_path=train_path, 
                                label_column=self.label_class,
                                dropout=dropout, 
                                batch_size=batch_size, 
                                learning_rate=learning_rate,
                                optim_name=optim_name, 
                                crit_name="CELoss + MSELoss")
                            
        if visualize:
            self.visualize_metrics(metrics)
            return None
        return metrics
    
    def visualize_metrics(self, metrics):
        self.mtx.confus_matrix(last_labels=metrics['all_labels'][-1],
                               last_preds=metrics['all_preds'][-1],
                               label_encoder=self.label_encoder.classes_,
                               save_folder_path=self.new_folder_path,
                               plt_name="confusion_matrix_last_model")
        
        self.mtx.train_val_loss(epoch=metrics['epoch'], 
                                train_loss=metrics['train_loss'],
                                val_loss=metrics['val_loss'], 
                                save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_acc(epoch=metrics['epoch'], 
                               train_acc=metrics['train_acc'],
                               val_acc=metrics['val_acc'], 
                               save_folder_path=self.new_folder_path)
        
        self.mtx.f1score(epoch=metrics['epoch'], 
                         f1_score=metrics['f1_score'],
                         save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_mae(epoch=metrics['epoch'], 
                               train_mae=metrics['train_mae'],
                               val_mae=metrics['val_mae'], 
                               save_folder_path=self.new_folder_path)
        
        self.mtx.train_val_rmse(epoch=metrics['epoch'], 
                                train_rmse=metrics['train_rmse'],
                                val_rmse=metrics['val_rmse'], 
                                save_folder_path=self.new_folder_path)
        
        self.mtx.regression_scatter(metrics=metrics, 
                                    save_folder_path=self.new_folder_path)
    
    def predict(self, model_path, text_data=None, scalar_data=None):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = checkpoint['model']
        model.eval()
        label_encoder = checkpoint['label_encoder']
        scaler_reg = checkpoint.get('scaler_reg')
        
        X = _prepare_inference_data(
            text_data, scalar_data,
            checkpoint.get('vectorizer_text'), checkpoint.get('vectorizer_scalar'))
        
        with torch.no_grad():
            class_logits, reg_output = model(X)
            pred_idx = torch.argmax(class_logits, dim=1).item()
            predicted_class = label_encoder.inverse_transform([pred_idx])[0]
            pred_norm = reg_output.item()
            predicted_value = scaler_reg.inverse_transform([[pred_norm]])[0][0] if scaler_reg else pred_norm
        
        return predicted_class, float(predicted_value)

    def finetune(self,
                 pretrained_path,
                 train_path,
                 val_path=None,
                 num_epochs=10,
                 hidden_size=None,
                 num_layers=None,
                 batch_size=128,
                 learning_rate=0.0001,
                 dropout=None,
                 optim_name='Adam',
                 crit_cls_name='CELoss',
                 crit_reg_name='MSELoss',
                 device_name='cuda',
                 loss_ratio_cls=1.0,
                 loss_ratio_reg=1.0):
        
        """Дообучение мультимодели."""

        print(f"Загрузка модели из: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        self.model = checkpoint['model']
        hidden_size, num_layers, dropout = _finetune_arch_params(checkpoint, self.model, hidden_size, num_layers, dropout)
        self.vectorizer_text = checkpoint.get('vectorizer_text')
        self.vectorizer_scalar = checkpoint.get('vectorizer_scalar')
        self.label_encoder = checkpoint['label_encoder']
        self.scaler_reg = checkpoint.get('scaler_reg')
        self.num_classes = checkpoint['num_classes']
        
        self.train_path = train_path
        self.val_path = val_path
        self.df_train = pd.read_csv(train_path)
        
        cls_col = self._col(self.label_class)
        reg_col = self._col(self.label_reg)
        
        # Векторизация
        if val_path is None:
            features = []
            
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features.append(sparse.csr_matrix(feat))
            
            X_all = sparse.hstack(features).tocsr() if len(features) > 1 else features[0]
            
            y_class_all = self.label_encoder.transform(self.df_train[cls_col])
            y_reg_all = self.scaler_reg.transform(self.df_train[reg_col].values.reshape(-1, 1)).flatten() if self.scaler_reg else self.df_train[reg_col].values
            
            self.X_train, self.X_val, self.y_class_train, self.y_class_val, \
                self.y_reg_train, self.y_reg_val = train_test_split(
                    X_all, y_class_all, y_reg_all,
                    test_size=self.split_ratio, random_state=self.random_state)
        else:
            self.df_val = pd.read_csv(val_path)
            
            # Векторизация train
            features_tr = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_train[col].fillna(''))
                    features_tr.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_train[col].fillna(0).values.reshape(-1, 1))
                    features_tr.append(sparse.csr_matrix(feat))
            
            self.X_train = sparse.hstack(features_tr).tocsr() if len(features_tr) > 1 else features_tr[0]
            self.y_class_train = self.label_encoder.transform(self.df_train[cls_col])
            self.y_reg_train = self.scaler_reg.transform(self.df_train[reg_col].values.reshape(-1, 1)).flatten() if self.scaler_reg else self.df_train[reg_col].values
            
            # Векторизация val
            features_val = []
            if self.vectorizer_text is not None:
                vec_t = self.vectorizer_text if isinstance(self.vectorizer_text, list) else [self.vectorizer_text]
                cols_t = self.text_column if isinstance(self.text_column, list) else [self.text_column]
                for vec, col in zip(vec_t, cols_t):
                    feat = vec.transform(self.df_val[col].fillna(''))
                    features_val.append(feat)
            
            if self.vectorizer_scalar is not None:
                vec_s = self.vectorizer_scalar if isinstance(self.vectorizer_scalar, list) else [self.vectorizer_scalar]
                cols_s = self.data_column if isinstance(self.data_column, list) else [self.data_column]
                for vec, col in zip(vec_s, cols_s):
                    feat = vec.transform(self.df_val[col].fillna(0).values.reshape(-1, 1))
                    features_val.append(sparse.csr_matrix(feat))
            
            self.X_val = sparse.hstack(features_val).tocsr() if len(features_val) > 1 else features_val[0]
            self.y_class_val = self.label_encoder.transform(self.df_val[cls_col])
            self.y_reg_val = self.scaler_reg.transform(self.df_val[reg_col].values.reshape(-1, 1)).flatten() if self.scaler_reg else self.df_val[reg_col].values
        
        self.input_size = self.X_train.shape[1]

        saved_input_size = checkpoint.get('input_size')
        if saved_input_size and saved_input_size != self.input_size:
            raise ValueError(f"Несовпадение размерности! В модели: {saved_input_size}, в данных: {self.input_size}")
        
        self.mtx = Metrics()
        self.new_folder_path = dirruns(folder_name="finetune_")
        
        headers = ["epoch", "train_loss", "val_loss", "val_acc", "f1_score", "val_mae", "val_rmse"]
        self.csv_path = f"{self.new_folder_path}/result.csv"
        if not os.path.isfile(self.csv_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        # DataLoader
        train_dataset = MultiTaskDataset(self.X_train, self.y_class_train, self.y_reg_train)
        val_dataset = MultiTaskDataset(self.X_val, self.y_class_val, self.y_reg_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор
        crit_class = CLASS_CRITERIA.get(crit_cls_name, nn.CrossEntropyLoss)()
        crit_reg = REG_CRITERIA.get(crit_reg_name, nn.MSELoss)()
        optimizer = OPTIMIZERS.get(optim_name, optim.Adam)(self.model.parameters(), lr=learning_rate)
        device = DEVICES.get(device_name, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(device)
        print(f"Дообучение на устройстве: {device}")
        
        # Создание тренера
        trainer = MultiTaskTrainer(model=self.model,
                                   criterion_class=crit_class,
                                   criterion_reg=crit_reg,
                                   optimizer=optimizer,
                                   device=device,
                                   train_loader=train_loader,
                                   val_loader=val_loader,
                                   label_encoder=self.label_encoder,
                                   vectorizer_text=self.vectorizer_text,
                                   vectorizer_scalar=self.vectorizer_scalar,
                                   new_folder_path=self.new_folder_path,
                                   loss_ratio_cls=loss_ratio_cls,
                                   loss_ratio_reg=loss_ratio_reg,
                                   scaler_reg=self.scaler_reg)
        
        metrics = trainer.train(num_epochs=num_epochs,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                input_size=self.input_size,
                                num_classes=self.num_classes,
                                train_path=train_path,
                                label_column=self.label_class,
                                dropout=dropout,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                optim_name=optim_name,
                                crit_name="CELoss + MSELoss")
        
        self.visualize_metrics(metrics)
        return

    def tune(self,
            model_name,
            train_path,
            val_path,
            iterations,
            num_epoch,
            hidden_size_range=[64, 128, 256],
            num_layers_range=[1, 2, 3],
            learning_rate_range=(1e-5, 1e-2),
            dropout_range=(0.0, 0.5),
            batch_size_range=[32, 64, 128],
            optim_name='Adam',
            crit_cls_name='CELoss',         
            crit_reg_name='MSELoss',         
            device_name='cuda',
            loss_ratio_cls=1.0,              
            loss_ratio_reg=1.0,              
            seed=None):
        """
        Подбор оптимальных гиперпараметров с использованием Optuna.
        """
        
        warnings.filterwarnings("ignore", message=".*dropout option adds dropout after all but last recurrent layer.*")
        warnings.filterwarnings("ignore", category=UserWarning)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.num_epoch = num_epoch
        self.optim_name = optim_name
        self.crit_cls_name = crit_cls_name        
        self.crit_reg_name = crit_reg_name        
        self.device_name = device_name
        self.loss_ratio_cls = loss_ratio_cls      
        self.loss_ratio_reg = loss_ratio_reg      
        self.tune_folder_path = dirruns("tune_")
        
        headers = ["iteration", "min_val_loss", "hidden_size", "num_layers", 
                "batch_size", "learning_rate", "dropout"]
        self.csv_tune_path = f"{self.tune_folder_path}/result_tune.csv"
        if not os.path.isfile(self.csv_tune_path):
            pd.DataFrame(columns=headers).to_csv(self.csv_tune_path, index=False)
        
        if seed is not None:
            sampler = TPESampler(seed=seed)
            print(f"Используется фиксированный seed={seed}")
        else:
            sampler = TPESampler()
            print(f"Используется случайный seed")
        
        study = optuna.create_study(study_name="lancetnic_multitask_tune",
                                    direction="minimize",
                                    sampler=sampler)
        
        pbar = tqdm(total=iterations, desc="Оптимизация гиперпараметров", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", hidden_size_range)
            num_layers = trial.suggest_categorical("num_layers", num_layers_range)
            learning_rate = trial.suggest_float("learning_rate", 
                                            learning_rate_range[0], 
                                            learning_rate_range[1], 
                                            log=True)
            dropout = trial.suggest_float("dropout", 
                                        dropout_range[0], 
                                        dropout_range[1])
            batch_size = trial.suggest_categorical("batch_size", batch_size_range)
            
            try:
                tuning = self.train(model_name=self.model_name,
                                    train_path=self.train_path,
                                    val_path=self.val_path,
                                    num_epochs=self.num_epoch,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    dropout=dropout,
                                    optim_name=self.optim_name,
                                    crit_cls_name=self.crit_cls_name,     
                                    crit_reg_name=self.crit_reg_name,     
                                    device_name=self.device_name,
                                    loss_ratio_cls=self.loss_ratio_cls,   
                                    loss_ratio_reg=self.loss_ratio_reg,   
                                    folder_name="tune_",
                                    visualize=False,
                                    save_folder_path=self.tune_folder_path)
                
                min_val_loss = min(tuning['val_loss'])
                
                # Сохранение результатов trial
                hyperparams = {"trial": trial.number + 1,
                               "model_params": {"hidden_size": hidden_size, 
                                                "num_layers": num_layers, "dropout": dropout},
                               "train_params": {"num_epochs": self.num_epoch, 
                                                "batch_size": batch_size, 
                                                "learning_rate": learning_rate, 
                                                "optimizer": self.optim_name, 
                                                "criterion_cls": self.crit_cls_name,    
                                                "criterion_reg": self.crit_reg_name,    
                                                "loss_ratio_cls": self.loss_ratio_cls,  
                                                "loss_ratio_reg": self.loss_ratio_reg}, 
                               "result": {"min_val_loss": min_val_loss}}
                
                trials_dir = f"{self.tune_folder_path}/trials"
                os.makedirs(trials_dir, exist_ok=True)

                path_config = f"{trials_dir}/tune_hyperparams_trial_{trial.number + 1}.yaml"
                with open(path_config, 'w', encoding='utf-8') as f:
                    yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
                
                row = {"iteration": trial.number + 1,
                       "min_val_loss": f"{min_val_loss:.4f}",
                       "hidden_size": hidden_size,
                       "num_layers": num_layers,
                       "batch_size": batch_size,
                       "learning_rate": f"{learning_rate:.2e}",
                       "dropout": f"{dropout:.3f}"}
                
                pd.DataFrame([row], columns=headers).to_csv(self.csv_tune_path, mode='a', header=False, index=False)
                
                return min_val_loss
                
            except Exception as e:
                print(f"\n✗ Ошибка в trial {trial.number + 1}: {e}")
                return float('inf')
        
        def progress_callback(study, trial):
            pbar.update(1)
            if study.best_trial:
                pbar.set_postfix({'best_loss': f'{study.best_trial.value:.4f}'})

        try:
            study.optimize(objective, n_trials=iterations, show_progress_bar=False, callbacks=[progress_callback])
        except KeyboardInterrupt:
            print("\n⚠️ Оптимизация прервана пользователем.")
        finally:
            pbar.close()
        
        if not study.trials:
            print("Не удалось завершить ни один trial.")
            return None

        best_trial = study.best_trial
        
        print("\n" + "="*70)
        print("🏆 ПОДБОР ЗАВЕРШЕН!")
        print("="*70)
        print(f"Лучший trial: #{best_trial.number + 1}")
        print(f"Лучший min_val_loss: {best_trial.value:.4f}")
        print("Лучшие параметры:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("="*70)
        
        best_params = best_trial.params.copy()
        best_params['num_epochs'] = self.num_epoch
        best_params['optimizer'] = self.optim_name
        best_params['criterion_cls'] = self.crit_cls_name      
        best_params['criterion_reg'] = self.crit_reg_name      
        best_params['loss_ratio_cls'] = self.loss_ratio_cls    
        best_params['loss_ratio_reg'] = self.loss_ratio_reg    
        
        best_config = {"best_trial_number": best_trial.number + 1,
                       "best_val_loss": best_trial.value,
                       "best_parameters": best_params,
                       "seed_used": seed}
        
        best_path = f"{self.tune_folder_path}/best_hyperparams.yaml"
        with open(best_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, indent=2)
        
        print(f"\n📁 Лучшие параметры: {best_path}")
        print(f"📊 Все trials: {self.csv_tune_path}")
        
        return best_params

    