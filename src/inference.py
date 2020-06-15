## Inspired by Joeri Hermans' https://github.com/montefiore-ai/hypothesis
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from nets import ResNet
from train import make_seed


class ConditionalResNetRatioEstimator(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int = 64, num_pred_classes: int = 1
    ):
        super().__init__()

        self.net = ResNet(in_channels, mid_channels, num_pred_classes)

    def forward(self, inputs, outputs):
        log_ratio = self.log_ratio(inputs, outputs)
        return log_ratio.sigmoid(), log_ratio

    def log_ratio(self, inputs, outputs):
        return self.net(torch.cat([inputs, outputs], dim=1))


class ConditionalRatioEstimator(nn.Module):
    def __init__(self, estimator, batch_size, device="cpu") -> None:
        super().__init__()

        self.estimator = estimator
        self.criterion = nn.BCELoss().to(device)

        chunked_batch_size = batch_size // 2
        self._ones = torch.ones(chunked_batch_size).to(device)
        self._zeros = torch.zeros(chunked_batch_size).to(device)

    def _compute_loss(self, inputs, outputs):
        in_a, in_b = inputs.chunk(2)
        out_a, out_b = outputs.chunk(2)

        y_dep_a, _ = self.estimator(in_a, out_a)
        y_indep_a, _ = self.estimator(in_a, out_b)
        y_dep_b, _ = self.estimator(in_b, out_b)
        y_indep_b, _ = self.estimator(in_b, out_a)

        loss_a = self.criterion(y_dep_a, self.ones) + self.criterion(
            y_indep_a, self.zeros
        )
        loss_b = self.criterion(y_dep_b, self.ones) + self.criterion(
            y_indep_b, self.zeros
        )

        return loss_a + loss_b

    def forward(self, inputs, outputs):
        return self._compute_loss(inputs, outputs)


class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cpu") -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.train_loss = []
        self.val_loss = []

    def fit(
        self,
        n_epochs: int,
        learning_rate: float = 0.001,
        early_stop_patience: int = 10,
        lr_patience: int = 2,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=lr_patience, verbose=True
        )
        best_val_score = np.inf
        patience_counter = 0
        best_state_dict = None

        for epoch in range(n_epochs):
            self.model.train()

            train_loss = []
            for _, inputs, _, outputs in self.train_loader:
                inputs, outputs = inputs.to(self.device), outputs.to(device)
                optimizer.zero_grad()
                loss = self.model(inputs, outputs)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(train_loss))

            val_loss = []
            with torch.no_grad():
                self.model.eval()
                for _, inputs, _ outputs in self.val_loader:
                    inputs, outputs = inputs.to(self.device), outputs.to(device)
                    loss = self.model(inputs, outputs)
                    val_loss.append(loss.item())
            self.val_loss.append(np.mean(val_loss))

            lr_scheduler.step(self.val_loss[-1])

            val_loss_string = colored(
                f"{self.val_loss[-1]:.3f}", "red" if self.val_loss[-1] > best_val_score else "green"
            )

            print(
                f"Epoch {epoch + 1:2d}: "
                f"Train loss: {self.train_loss[-1]:.3f, "
                f"Val loss: {val_loss_string}"
            )

            if self.val_loss[-1] < best_val_score:
                best_val_score = self.val_loss[-1]
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == early_stop_patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(best_state_dict)
                    print("Early stopping...")
                    return self
        return self

    def save_model(self, path: Path) -> Path:
        model_dict = {
            "state_dict": self.model.state_dict()
        }
        torch.save(model_dict, f'../results/{path}.pt')
        return path
