# Training Hyperparameters (train method)

## Class TextClass()

| Attribute       | Type      | Values         | Description                                                                 |
|----------------|----------|----------------|--------------------------------------------------------------------------|
| `text_column`  | string   | Any valid column name | Name of the column containing text data in the dataset                |
| `label_column` | string   | Any valid column name | Name of the column containing target labels                           |
| `split_ratio`  | float    | (0.01, 0.99)   | Proportion of the dataset to use for validation (when val_path=None)  |
| `random_state` | int      | (1, 99)        | Random seed for reproducible splits (default: 42)                     |

## Class TextScalarClass()

| Attribute       | Type      | Values         | Description                                                                 |
|----------------|----------|----------------|--------------------------------------------------------------------------|
| `text_column`  | string   | Any valid column name or None | Name of column with text data (None for numeric only)             |
| `data_column`  | list     | List of column names | Names of columns containing numerical features                      |
| `label_column` | string   | Any valid column name | Name of the column containing target labels                           |
| `split_ratio`  | float    | (0.01, 0.99)   | Proportion of the dataset to use for validation (when val_path=None)  |
| `random_state` | int      | (1, 99)        | Random seed for reproducible splits (default: 42)                     |

## Methods TextClass() and TextScalarClass()

### `train`

| Parameter       | Type      | Values                     | Description                                                                 |
|----------------|----------|------------------------------|--------------------------------------------------------------------------|
| `model_name`   | class    | `LancetMC`, `LancetMCA`      | Model architecture class (LancetMC for text, LancetMCA for mixed)        |
| `train_path`   | string   | Absolute/relative path       | Path to training dataset in CSV format                                  |
| `val_path`     | string   | Path or `None`               | Path to validation dataset. If None, splits from train data             |
| `num_epochs`   | int      | (1, ...)                    | Number of complete passes through training data                         |
| `hidden_size`  | int      | (1, ...)                    | Number of neurons in each hidden layer (default: 256)                   |
| `num_layers`   | int      | (1, ...)                    | Depth of the model (number of hidden layers, default: 1)                |
| `batch_size`   | int      | (1, ...)                    | Number of samples per gradient update (default: 128)                    |
| `learning_rate`| float    | (1e-5, 1e-1)                | Step size for optimizer (default: 0.001)                                |
| `dropout`      | float    | (0.0, 0.99)                 | Probability of neuron dropout for regularization (default: 0)           |
| `optim_name`   | string   | 'Adam', 'RAdam', 'SGD', 'RMSprop', 'Adadelta' | Optimization algorithm (default: 'Adam')         |
| `crit_name`    | string   | 'CELoss', 'BCELoss'         | Loss function ('CELoss' for multi-class, 'BCELoss' for binary)         |

### `predict`

| Parameter       | Type      | Values                     | Description                                                                 |
|----------------|----------|------------------------------|--------------------------------------------------------------------------|
| `model_path`   | string   | Absolute/relative path       | Path to saved model checkpoint file (.pth)                              |
| `text`         | string   | Any text input               | Text to classify (for TextClass)                                        |
| `numeric`      | list     | List of float values         | Numerical features for prediction (TextScalarClass only)                |



# Model Architectures

## LancetMC (LSTM-based Multiclass Classifier)

A standard sequential model for multiclass classification using LSTM layers. Designed primarily for text classification tasks where temporal dependencies in the data need to be captured.

## LancetMCA (LSTM with Attention Multiclass Classifier)

An enhanced version of LancetMC incorporating attention mechanisms. Designed for more complex classification tasks where certain parts of the input sequence are more important than others.

| Key Differences Between Models | LancetMC          | LancetMCA                     |
|--------------------------------|-------------------|-------------------------------|
| Feature                        |                   |                               |
| Core Architecture              | Basic LSTM        | LSTM + Attention              |
| Complexity                     | Lower             | Higher                        |
| Computational Cost             | Less resource-intensive | More resource-intensive  |
| Best For                       | Pure text classification | Mixed data or complex patterns |
| Interpretability               | Standard          | Provides attention weights    |
| Sequence Handling              | Good              | Excellent for long sequences  |