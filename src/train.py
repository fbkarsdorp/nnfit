import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from nets import FCN
from dataset import SimulationData, worker_init_fn, collate


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loss = []
        self.val_loss = []
        self.val_scores = []

    def fit(
        self, n_epochs: int, learning_rate: float = 0.01, patience: int = 10
    ) -> "Trainer":
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
                    outputs, labels.unsqueeze(-1).float(), reduction="mean"
                )
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(train_loss))

            val_loss, y_true, y_pred = [], [], []
            self.model.eval()
            for labels, inputs in self.val_loader:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = F.binary_cross_entropy_with_logits(
                        outputs, labels.unsqueeze(-1).float(), reduction="mean"
                    )
                    val_loss.append(loss.item())
                    preds = (torch.sigmoid(outputs).numpy() > 0.5).tolist()
                    y_pred.extend(preds)
                    y_true.extend(labels.numpy().tolist())

            self.val_scores.append(accuracy_score(y_true, y_pred))
            self.val_loss.append(np.mean(val_loss))

            print(
                f"Epoch: {epoch + 1}, "
                f"Train loss: {self.train_loss[-1]:.3f}, "
                f"Val loss: {self.val_loss[-1]:.3f}, "
                f"Val accuracy: {self.val_scores[-1]:.3f}"
            )

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
            "model": {
                "model_class": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
            }
        }
        torch.save(save_dict, savepath)
        return savepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_individuals",
        type=int,
        default=1000,
        help="Number of individuals/datapoints per generation.",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=1000,
        help="Number of simulations per epoch.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.5,
        help="Relative start frequency of trait."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of simulations per batch."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of simulations to use for validation.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=25,
        help="Number of binning solutions."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200,
        help="Number of timesteps/generations to run the simulation.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the Neural Networks."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate forthe Adam optimizer."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of allowed epochs without improvement."
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of data loader workers."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed number."
    )

    args = parser.parse_args()

    train_data = SimulationData(
        start=args.start,
        nbins=args.n_bins,
        n_simulations=args.n_simulations,
        n_individuals=args.n_individuals,
        timesteps=args.timesteps,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate,
        worker_init_fn=worker_init_fn
        num_workers=args.n_workers
        drop_last=True
    )

    val_data = SimulationData(
        start=args.start,
        nbins=args.n_bins,
        n_simulations=int(args.val_size * args.n_simulations),
        n_individuals=args.n_individuals,
        timesteps=args.timesteps,
        seed=args.seed,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=collate,
        worker_init_fn=worker_init_fn
        num_workers=args.n_workers
        drop_last=True
    )

    model = FCN(args.timesteps)
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(args.n_epochs, learning_rate=args.learning_rate, patience=args.patience)
