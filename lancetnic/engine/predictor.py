import torch

class Predictor:
    def __init__(self):
        self.model_path=None
        self.text=None
        self.checkpoint=None
        self.model=None
        
    def predict(self, model_path, text):
        
        self.model_path=f"{model_path}"
        self.text=text
        # Загружаем на CPU. Так как векторизация в базовом трейне была через библиотеку sklearn, то только CPU!!!
        self.checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)  
        self.model = self.checkpoint['model'] 
        self.model.eval()  

        
        
        X = self.checkpoint['vectorizer'].transform([self.text]).toarray()
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            self.pred = torch.argmax(self.model(X), dim=1).item()
            self.class_name = self.checkpoint['label_encoder'].inverse_transform([self.pred])[0]

        return self.class_name