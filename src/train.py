import argparse
import json
import os
import uuid

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report
from termcolor import colored

from dataset import DataLoader
from nets import FCN, LSTMFCN, ResNet, InceptionTime
from test import plot_parameter_sweep, test_model
from test import evaluate_results, generate_test_samples, plot_fp_scores
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

    def fit(
        self,
        n_epochs: int,
        learning_rate: float = 0.001,
        early_stop_patience: int = 10,
        lr_patience: int = 2,
        evaluation_maximum: float = 1.0,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=lr_patience, verbose=True)
        best_val_score = -np.inf
        patience_counter = 0
        best_state_dict = None

        processed_bins = []
        for epoch in range(n_epochs):
            self.model.train()

            train_loss = []
            for labels, _, bs, inputs in self.train_loader:
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
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                processed_bins.extend(bs.numpy().tolist())
            self.train_loss.append(np.mean(train_loss))

            val_loss, val_results = [], []
            with torch.no_grad():
                self.model.eval()
                for labels, ss, bs, inputs in self.val_loader:
                    labels, inputs = (
                        labels.to(self.device),
                        inputs.to(self.device).unsqueeze(1),
                    )
                    outputs = self.model(inputs)
                    loss = F.binary_cross_entropy_with_logits(
                        outputs, labels.unsqueeze(-1).float()
                    )
                    val_loss.append(loss.item())
                    preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5)
                    preds = preds.squeeze(1).astype(int).tolist()
                    labels = labels.cpu().numpy().tolist()
                    for i in range(len(preds)):
                        val_results.append({
                            "bin": bs[i].item(),
                            "selection": ss[i].item(),
                            "correct": int(preds[i] == labels[i]),
                            "y_true": labels[i],
                            "y_pred": preds[i],
                        })

            df = pd.DataFrame(val_results)
            self.val_scores.append(accuracy_score(
                *df.loc[df['selection'] <= evaluation_maximum, ['y_true', 'y_pred']].values.T))
            self.val_loss.append(np.mean(val_loss))

            lr_scheduler.step(self.val_scores[-1])

            val_accuracy_string = colored(
                f"{self.val_scores[-1]:.3f}", "red" if self.val_scores[-1] < best_val_score else "green")

            print(
                f"Epoch {epoch + 1:2d}: "
                f"Train loss: {self.train_loss[-1]:.3f}, "
                f"Val loss: {self.val_loss[-1]:.3f}, "
                f"Val accuracy: {val_accuracy_string}"
            )
            
            if self.val_scores[-1] > best_val_score:
                best_val_score = self.val_scores[-1]
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

    def save_model(self, savepath: Path) -> Path:
        model_dict = {
            "args": self.model.input_args,
            "state_dict": self.model.state_dict()
        }
        torch.save(model_dict, f'../results/{savepath}.pt')
        return savepath


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
        compute_fiv=args.compute_frequency_increment_values,
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
        compute_fiv=args.compute_frequency_increment_values,
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

    if args.model == "FCN":
        model = FCN(1, 1)
    elif args.model == "LSTMFCN":
        model = LSTMFCN(args.hidden_size, 1, args.num_layers, 1,
                        args.dropout, args.rnn_dropout, args.bidirectional)
    elif args.model == "INCEPTION":
        model = InceptionTime(1, 1)
    else:
        model = ResNet(1)

    model = model.to(device)
    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.fit(args.n_epochs, learning_rate=args.learning_rate,
                lr_patience=args.learning_rate_patience, early_stop_patience=args.early_stop_patience,
                evaluation_maximum=args.evaluation_maximum)

    run_id = str(uuid.uuid1())[:8] if args.outfile is None else args.outfile

    trainer.model.eval()
    trainer.save_model(run_id)

    if args.test:
        print("Evaluating the model...")
        if args.test_samples is not None:
            print("Using precomputed test-samples...")
        else:
            print("Generating test samples...")
        test_samples = args.test_samples if args.test_samples is not None else generate_test_samples(
            args.timesteps, args.n_agents, args.n_workers
        )
            
        nn_results, fit_results = test_model(
            model, test_samples, args.timesteps, args.n_agents,
            device=device, normalize_samples=args.normalize_samples
        )
        nn_scores = evaluate_results(nn_results, prefix="NN_")
        fit_scores = evaluate_results(fit_results, prefix="FIT_")
        
        plot_parameter_sweep(fit_results, nn_results).savefig(
            f"../results/{run_id}_params_sweep.png", dpi=300)
        plot_fp_scores(fit_results, nn_results).savefig(
            f"../results/{run_id}_fp_scores.png", dpi=300)
        
        return run_id, nn_scores, fit_scores
        

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
        "--model",
        type=str,
        choices=("FCN", "LSTMFCN", "RESNET", "INCEPTION"),
        default="FCN",
        help="Neural architecture for training."
    )
    parser.add_argument(
        "--normalize_samples",
        action="store_true",
        help="Scale samples to unit norm."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of LSTM network"
    )
    parser.add_argument(
        "--evaluation_maximum",
        type=float,
        default=1,
        help="Maximum selection strength used for validation evaluation."
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
    parser.add_argument(
        "--test", action="store_true", help="Apply models to linspace test set."
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
