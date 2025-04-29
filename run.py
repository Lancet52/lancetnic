from models.lancet_binary import LancetBC
from train import Trainer
    
trainer = Trainer(
        model_name=LancetBC,
        train_path="datasets/spam_train.csv",
        val_path="datasets/spam_val.csv"
    )
    
trainer.load_data()
trainer.setup_training(hidden_size=256, num_layers=1)
trainer.train(num_epochs=100)