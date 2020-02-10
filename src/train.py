from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from dataset import KarjusData, WrightFisherData, worker_init_fn, collate


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loss = []
        self.val_loss = []
        self.val_scores = []
        
    def fit(self, n_epochs: int, learning_rate: float = 0.01, patience: int = 10) -> 'Trainer':
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

        self.model.train()
        for epoch in range(n_epochs):

            train_loss = []
            for labels, inputs in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy_with_logits(
                    outputs, labels.unsqueeze(-1).float(), reduction='mean')
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(train_loss))

            val_loss, y_true, y_pred = [], [], []
            self.model.eval()
            for labels, inputs = self.val_loader:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = F.binary_cross_entropy_with_logits(
                        outputs, labels.unsqueeze(-1).float(), reduction='mean')
                    val_loss.append(loss.item())
                    preds = (torch.sigmoid(outputs).numpy() > 0.5).tolist()
                    y_pred.extend(preds)
                    y_true.extend(labels.numpy().tolist())
                    
            self.val_scores.append(accuracy_score(y_true, y_pred))
            self.val_loss.append(np.mean(val_loss))

            print(f'Epoch: {epoch + 1}, '
                  f'Train loss: {self.train_loss[-1]:.3f}, '
                  f'Val loss: {self.val_loss[-1]:.3f}, '
                  f'Val accuracy: {self.val_scores[-1]:.3f}')

            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(best_state_dict)
                    print("Early stopping...")
                    return self
        return self

    def save_model(self, savepath: Path) -> Path:
        save_dict = {
            'model': {
                'model_class': self.model.__class__.__name__,
                'state_dict': self.model.state_dict()
            }
        }
        torch.save(save_dict, savepath)
        return savepath
