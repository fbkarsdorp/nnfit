## Adapted from Joeri Hermans' https://github.com/montefiore-ai/hypothesis

import argparse
import uuid
from pathlib import Path

from termcolor import colored

import numpy as np
import torch
import torch.nn as nn

from dataset import DataLoader
from nets import ResNet
from train import make_seed
from utils import Distorter

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
        return self.net(torch.cat((inputs.view(-1, 1, 1), outputs), dim=2))


class ConditionalRatioEstimator(nn.Module):
    def __init__(self, estimator, batch_size, device="cpu") -> None:
        super().__init__()

        self.estimator = estimator
        self.criterion = nn.BCELoss().to(device)

        chunked_batch_size = batch_size // 2
        self._ones = torch.ones(chunked_batch_size).to(device).unsqueeze(1)
        self._zeros = torch.zeros(chunked_batch_size).to(device).unsqueeze(1)

    def _compute_loss(self, inputs, outputs):
        in_a, in_b = inputs.chunk(2)
        out_a, out_b = outputs.chunk(2)

        y_dep_a, _ = self.estimator(in_a, out_a)
        y_indep_a, _ = self.estimator(in_a, out_b)
        y_dep_b, _ = self.estimator(in_b, out_b)
        y_indep_b, _ = self.estimator(in_b, out_a)

        loss_a = self.criterion(y_dep_a, self._ones) + self.criterion(
            y_indep_a, self._zeros
        )
        loss_b = self.criterion(y_dep_b, self._ones) + self.criterion(
            y_indep_b, self._zeros
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
                inputs, outputs = inputs.to(self.device), outputs.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                loss = self.model(inputs, outputs)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(train_loss))

            val_loss = []
            with torch.no_grad():
                self.model.eval()
                for _, inputs, _, outputs in self.val_loader:
                    inputs, outputs = inputs.to(self.device), outputs.to(self.device).unsqueeze(1)
                    loss = self.model(inputs, outputs)
                    val_loss.append(loss.item())
            self.val_loss.append(np.mean(val_loss))

            lr_scheduler.step(self.val_loss[-1])

            val_loss_string = colored(
                f"{self.val_loss[-1]:.3f}", "red" if self.val_loss[-1] > best_val_score else "green"
            )

            print(
                f"Epoch {epoch + 1:2d}: "
                f"Train loss: {self.train_loss[-1]:.3f}, "
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


def run_experiment(args):
    train_distortion, val_distortion = None, None
    if args.distortion:
        train_distortion = Distorter(loc=0, sd=args.distortion_sd, seed=args.seed)
        val_distortion = Distorter(loc=0, sd=args.distortion_sd, seed=args.seed + 1)

    train_params = dict(
        distortion=train_distortion,
        variable_binning=args.variable_binning,
        start=args.start,
        varying_start_value=args.varying_start_value,
        n_agents=args.n_agents,
        timesteps=args.timesteps,
        compute_fiv=False,
    )

    train_loader = DataLoader(
        train_params,
        batch_size=args.batch_size,
        seed=args.seed,
        distortion=args.distortion,
        n_sims=args.n_sims,
        train=True,
        n_workers=args.n_workers,
        normalize_samples=args.normalize_samples,
    )

    val_params = dict(
        distortion=val_distortion,
        variable_binning=args.variable_binning,
        start=args.start,
        varying_start_value=args.varying_start_value,
        n_agents=args.n_agents,
        timesteps=args.timesteps,
        compute_fiv=False
    )
    
    val_loader = DataLoader(
        val_params,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        distortion=args.distortion,
        n_sims=int(args.val_size * args.n_sims),
        train=False,
        n_workers=args.n_workers,
        normalize_samples=args.normalize_samples,
    )

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")

    estimator = ConditionalResNetRatioEstimator(1)
    model = ConditionalRatioEstimator(estimator, args.batch_size, device=device)

    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.fit(args.n_epochs, learning_rate=args.learning_rate,
                lr_patience=args.learning_rate_patience, early_stop_patience=args.early_stop_patience)

    run_id = str(uuid.uuid1())[:8] if args.outfile is None else args.outfile

    trainer.model.eval()
    trainer.save_model(run_id)
    

def get_arguments():
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
        "--varying_start_value", action="store_true",
        help="Varying relative start frequency of traits."
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
        "--normalize_samples",
        action="store_true",
        help="Scale samples to unit norm."
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
        type=float,
        nargs=2,
        default=None,
        help="Apply distortion to wright fisher simulations sampled from d_i ~ N(loc, sd).",
    )
    parser.add_argument(
        "--distortion_sd",
        type=float,
        default=0.1,
        help="Distortion values sampled from N(0, distortion_sd)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--learning_rate_patience",
        type=int,
        default=3,
        help="Number of epochs without improvement before reducing the learning rate."
    )
    parser.add_argument("--cuda", action="store_true", help="Use Cuda.")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Number of allowed epochs without improvement.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1, help="Number of data loader workers."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed number.")
    parser.add_argument("--outfile", type=str, default=None)

    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = make_seed()

    return args

if __name__ == '__main__':
    args = get_arguments()
    torch.manual_seed(args.seed)
    run_experiment(args)
