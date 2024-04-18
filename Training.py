from models import *
import DataPrepering
from torch.utils.data import  DataLoader, TensorDataset
import torch.optim as optim

class Trainer():
    def __init__(self):


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_to_train_embeddings = EmmbendingModel(num_champs=1000, embed_dim=200, num_heads=8).to(self.device)
        self.model_to_train_predictions = PredictingModel(num_champs=1000, embed_dim=200, num_heads=8).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None


    def train_embeddings(self, path_to_data):
        champions, context, results = DataPrepering.create_tensors_for_embeddings(path_to_data, self.device)
        matches = TensorDataset(champions, context, results)
        dataloader = DataLoader(matches, batch_size=1024, shuffle=True)
        self.optimizer = optim.AdamW(self.model_to_train_embeddings.parameters(), lr=0.001)
        for epoch in range(60):
            for inputs1, inputs2, targets in dataloader:
                self.model_to_train_embeddings.train()
                outputs = self.model_to_train_embeddings(inputs1, inputs2)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train_predictions(self, path_to_data, results):
        champions, results = DataPrepering.create_tensors(path_to_data, results, self.device)
        matches = TensorDataset(champions, results)
        dataloader = DataLoader(matches, batch_size=1024, shuffle=True)
        self.optimizer = optim.AdamW(self.model_to_train_predictions.parameters(), lr=0.001)
        for epoch in range(60):
            for inputs1, targets in dataloader:
                self.model_to_train_predictions.train()
                outputs = self.model_to_train_predictions(inputs1)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

