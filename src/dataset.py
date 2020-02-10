import numpy as np
import pandas as pd
import scipy.stats as stats
import torch.utils.data

from simulation import EvoModel
from utils import Prior


def distort(trait_fd, r=0.5, r_sd=0):
    traits = np.arange(trait_fd.shape[0])
    new_trait_fd = np.zeros_like(trait_fd)
    for i, fd in enumerate(trait_fd):
        N = int(np.random.normal(loc=r, scale=r_sd) * fd.sum())
        population = np.random.choice(traits, size=N, replace=True, p=fd / fd.sum())
        for trait in population:
            new_trait_fd[i, trait] += 1
    return new_trait_fd


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size : (worker_id + 1) * split_size]


def collate(data):
    inputs, outputs = zip(*data)
    lengths = torch.LongTensor([output.size(1) for output in outputs])
    length = lengths.max().item()
    padded_outputs = []
    for i, output in enumerate(outputs):
        padded_outputs.append(
            F.pad(output, (0, length - output.size(1)), "constant", 0)
        )
    return inputs, torch.stack(padded_outputs)


class KarjusData(torch.utils.data.IterableDataset):
    def __init__(
        self,
        selection_prior=None,
        start=0.5,
        nbins=25,
        n_simulations=1000,
        n_individuals=1000,
        timesteps=200,
        seed=None,
    ):
        if selection_prior is None:
            selection_prior = stats.beta(1, 5)
        self.selection_prior = selection_prior
        self.data = np.arange(n_simulations)
        self.start = int(start * n_individuals)
        self.n_individuals = n_individuals
        self.timesteps = timesteps
        self.rnd = np.random.RandomState(seed)
        self.bins = np.array([x[0] for x in np.array_split(np.arange(4, self.timesteps), nbins)])
        self.bins[-1] = timesteps

    def __iter__(self):
        self.n_samples = 0
        return self

    def __next__(self):
        if self.n_samples < len(self.data):
            self.n_samples += 1
            s = self.selection_prior.rvs()
            if self.rnd.rand() < 0.5:
                s = 0
            j = np.zeros(self.timesteps)
            j[0] = self.start
            for i in range(1, self.timesteps):
                p_star = j[i - 1] * (1 + s) / (j[i - 1] * s + self.n_individuals)
                j[i] = self.rnd.binomial(self.n_individuals, min(p_star, 1))
            biased = int(s != 0)
            n_bins = self.rnd.choice(self.bins)
            binning = np.array_split(np.arange(self.timesteps), n_bins)
            j = np.array([j[ii].sum() for ii in binning])
            return {
                "bias": biased,
                "selection": s,
                "bins": n_bins,
                "data": j / np.array([self.n_individuals * len(b) for b in binning])
            }
        raise StopIteration


class WrightFisherData(torch.utils.data.IterableDataset):
    def __init__(
        self,
        n_simulations=1000,
        n_individuals=1000,
        timesteps=10,
        burn_in=1000,
        prior=None,
        r=0.5,
        r_sd=0,
        seed=None,
        n_workers=1,
        verbose=False,
    ):
        self.n_individuals = n_individuals
        self.data = np.arange(n_simulations)
        self.timesteps = timesteps
        self.burn_in = burn_in
        self.r = r
        self.r_sd = r_sd
        self.seed = seed
        self.verbose = verbose
        self.n_workers = n_workers
        if prior is None:
            prior = Prior(b=stats.uniform(-0.5, 1), mu=stats.uniform(0.0001, 0.1))
        self.prior = prior

    def __iter__(self):
        self.n_samples = 0
        return self

    def __next__(self):
        if self.n_samples < len(self.data):
            self.n_samples += 1
            params = self.prior.rvs()
            if self.rnd.rand() < 0.5:
                params["b"] = 0
            model = EvoModel(
                self.n_individuals,
                mu=params["mu"],
                b=params["b"],
                burn_in=self.burn_in,
                timesteps=self.timsteps,
                seed=self.seed,
            ).run()
            biased = int(params["b"] != 0)
            return biased, distort(model.trait_fd, r=self.r, r_sd=self.r_sd)
        raise StopIteration
