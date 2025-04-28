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
        