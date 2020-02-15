import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

from nets import FCN, LSTMFCN
from dataset import SimulationData, TestSimulationData, DataLoader
from utils import Distorter


def make_seed():
    now = datetime.now()
    seed = now.hour * 10000 + now.minute * 100 + now.second
    return seed


class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cpu") -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_loss = []
        self.val_loss = []
        self.val_scores = []

    def fit(self, n_epochs: int, learning_rate: float = 0.01, patience: int = 10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

        self.model.train()
        for epoch in range(n_epochs):

            train_loss = []
            for labels, _, _, inputs in self.train_loader:
                labels, inputs = (
                    labels.to(self.device),
                    inputs.to(self.device).unsqueeze(1),
                )
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy_with_logits(
                    outputs, labels.unsqueeze(-1).float()
                )
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(train_loss))

            val_loss, y_true, y_pred = [], [], []
            self.model.eval()
            for labels, _, _, inputs in self.val_loader:
                with torch.no_grad():
                    labels, inputs = (
                        labels.to(self.device),
                        inputs.to(self.device).unsqueeze(1),
                    )
                    outputs = self.model(inputs)
                    loss = F.binary_cross_entropy_with_logits(
                        outputs, labels.unsqueeze(-1).float()
                    )
                    val_loss.append(loss.item())
                    preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).tolist()
                    y_pred.extend(preds)
                    y_true.extend(labels.cpu().numpy().tolist())

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

    def evaluate(self, test_loader) -> Dict:
        self.model.eval()
        results = {"y_true": [], "y_pred": [], "probs": [], "selection": [], "bins": []}
        for labels, selection, bins, inputs in test_loader:
            with torch.no_grad():
                labels, inputs = (
                    labels.to(self.device),
                    inputs.to(self.device).unsqueeze(1),
                )
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).tolist()
                results["y_pred"].extend(preds)
                results["y_true"].extend(labels.cpu().numpy().tolist())
                results["selection"].extend(selection.numpy().tolist())
                results["bins"].extend(bins.numpy().tolist())
                results["probs"].extend(probs.tolist())
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_agents",
        type=int,
        default=1000,
        help="Number of individuals/datapoints per generation.",
    )
    parser.add_argument(
        "--n_sims", type=int, default=1000, help="Number of simulations per epoch.",
    )
    parser.add_argument(
        "--start", type=float, default=0.5, help="Relative start frequency of trait."
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Number of simulations per batch."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of simulations to use for validation.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200,
        help="Number of timesteps/generations to run the simulation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("FCN", "LSTMFCN"),
        default="FCN",
        help="Neural architecture for training."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of LSTM network"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in LSTM network"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout applied to time series."
    )
    parser.add_argument(
        "--rnn_dropout",
        type=float,
        default=0.0,
        help="Dropout applied between LSTM layers."
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use a bidirectional LSTM network."
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the Neural Networks.",
    )
    parser.add_argument(
        "--variable_binning", action="store_true", help="Apply variable binning."
    )
    parser.add_argument(
        "--distortion",
        action="store_true",
        help="Apply distortion to wright fisher simulations.",
    )
    parser.add_argument(
        "--compute_frequency_increment_values",
        action="store_true",
        help="Compute frequency increment values for the time series."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument("--cuda", action="store_true", help="Use Cuda.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of allowed epochs without improvement.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1, help="Number of data loader workers."
    )
    parser.add_argument(
        "--test", action="store_true", help="Apply models to linspace test set."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed number.")

    args = parser.parse_args()

    if args.seed is None:
        args.seed = make_seed()

    train_distortion, val_distortion = None, None
    if args.distortion:
        train_distortion = Distorter(seed=args.seed)
        val_distortion = Distorter(seed=args.seed + 1)

    train_data = SimulationData(
        distortion=train_distortion,
        variable_binning=args.variable_binning,
        start=args.start,
        n_sims=args.n_sims,
        n_agents=args.n_agents,
        timesteps=args.timesteps,
        seed=args.seed,
        compute_fiv=args.compute_frequency_increment_values,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size)

    val_data = SimulationData(
        distortion=val_distortion,
        variable_binning=args.variable_binning,
        start=args.start,
        n_sims=int(args.val_size * args.n_sims),
        n_agents=args.n_agents,
        timesteps=args.timesteps,
        seed=args.seed + 1,
        train=False,
        compute_fiv=args.compute_frequency_increment_values
    )

    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")

    if args.model == "FCN":
        model = FCN(1, 1).to(device)
    else:
        model = LSTMFCN(args.hidden_size, 1, args.num_layers, 1, args.dropout, args.rnn_dropout, args.bidirectional)
    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.fit(args.n_epochs, learning_rate=args.learning_rate, patience=args.patience)

    if args.test:
        test_distortion = None
        if args.distortion:
            test_distortion = Distorter(args.seed + 2)
        test_data = TestSimulationData(
            distorter=test_distortion,
            variable_binning=args.variable_binning,
            start=args.start,
            n_sims=args.n_sims,
            n_agents=args.n_agents,
            timesteps=args.timesteps,
            compute_fiv=args.compute_fiv
        )

        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        results = trainer.evaluate(test_loader)
