# Training Hyperparameters (train method)

## Class Binary()

| Attribute       | Type      | Values         | Description                                                                 |
|----------------|----------|----------------|--------------------------------------------------------------------------|
| `text_column`  | string   | description    | Name of the data column in the dataset                                   |
| `label_column` | string   | category       | Name of the label column in the dataset                                 |
| `split_ratio`  | float    | (0.01, 0.99)   | Proportion of the validation dataset                                    |
| `random_state` | int      | (1, 99)        | Seed for reproducible results (default: 42)                             |

## Methods

### `train`

| Parameter       | Type      | Values                     | Description                                                                 |
|----------------|----------|------------------------------|--------------------------------------------------------------------------|
| `model_name`   | class    | `LancetBC`                  | Reference to the model structure class                                   |
| `train_path`   | string   | Absolute, relative          | Path to the training dataset in CSV format                              |
| `val_path`     | string   | Absolute, relative, or `None` | Path to the validation dataset in CSV format. If not provided, set to `None` (default: splits from training data) |
| `num_epochs`   | int      | (1, ...)                    | Number of training epochs                                               |
| `hidden_size`  | int      | (1, ...)                    | Number of neurons in the model structure (default: 256)                 |
| `num_layers`   | int      | (1, ...)                    | Number of hidden layers in the model (default: 1)                       |
| `batch_size`   | int      | (1, ...)                    | Number of samples per batch (default: 128)                             |
| `learning_rate`| float    | (1e-5, 1e-1)                | Learning rate (default: 0.001)                                          |
| `dropout`      | float    | (0.0, 0.99)                 | Dropout rate for neurons during training (default: 0)                   |