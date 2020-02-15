import numpy as np
import scipy.stats as stats
import torch.utils.data
import torch.nn.functional as F

from typing import Tuple
from utils import apply_binning, Distorter, check_random_state
from simulation import wright_fisher
from fit import frequency_increment_values


class SimulationData:
    def __init__(
        self,
        selection_prior: Tuple[float, float] = (1, 5),
        distortion: Distorter = None,
        variable_binning=False,
        start: float = 0.5,
        n_sims: int = 1000,
        n_agents: int = 1000,
        timesteps: int = 200,
        seed: int = None,
        compute_fiv: bool = False,
        train: bool = True,
        min_bin_length: int = 4,
    ) -> None:

        self.rng = check_random_state(seed)
        self.selection_prior = stats.beta(selection_prior[0], selection_prior[1])
        self.distortion = distortion
        if variable_binning and self.distortion is None:
            raise ValueError("Variable binning requires distortion prior")
        self.variable = variable_binning
        self.data = np.arange(n_sims)
        self.start = start
        self.n_agents = n_agents
        self.timesteps = timesteps
        self.compute_fiv = compute_fiv
        self.train = train
        self.seed = seed
        self.n_samples = 0

        # sample bins following Karjus
        bins = {}
        maxlen = np.ceil(timesteps / min_bin_length)
        for i in range(1, int(maxlen + 1)):
            bins[i] = int(np.ceil(timesteps / i))
        bins = sorted(set(bins.values()))
        self.bins = np.array(bins)

        self.set_priors()

    def __len__(self):
        return self.data.shape[0]

    def set_priors(self) -> None:
        # Preset random values for reuse in validation
        n = len(self)
        self.selection_priors = self.selection_prior.rvs(n, random_state=self.rng)
        self.bias_priors = self.rng.rand(n)
        self.binnings = self.rng.choice(self.bins, size=n)

    def reset(self):
        self.n_samples = 0
        if not self.train:
            # Reinitialize Random Number Generator for consistent output
            self.rng = check_random_state(self.seed)
            # When using a distorter, reset it for consitent output in validation
            if self.distortion is not None:
                self.distortion.reset()
        if self.train:
            # For new pass over the data, we want new random settings
            self.set_priors()
        return self

    def next(self, n_bins=None):
        if self.n_samples < len(self.data):
            s = self.selection_priors[self.n_samples]
            if self.bias_priors[self.n_samples] < 0.5:
                s = 0

            j = wright_fisher(
                self.n_agents,
                self.timesteps,
                s,
                start=self.start,
                random_state=self.rng,
            )

            # binning
            if n_bins is None:
                n_bins = self.binnings[self.n_samples]
            j = apply_binning(
                j,
                n_bins,
                self.n_agents,
                variable=self.variable,
                distortions=distortions,
                random_state=self.rng,
            )

            if self.distortion is not None:
                j = self.distortion.distort(j)
                
            self.n_samples += 1

            biased = int(s != 0)

            if self.compute_fiv:
                j = frequency_increment_values(np.arange(1, len(j) + 1), j, clip=True)

            return biased, s, n_bins, torch.FloatTensor(j)

        raise StopIteration


class TestSimulationData(SimulationData):
    def __init__(
        self,
        distorter=None,
        variable_binning=False,
        start=0.5,
        n_sims=1000,
        n_bins=25,
        n_agents=1000,
        timesteps=200,
        compute_fiv=False,
    ) -> None:

        super().__init__(
            distortion=distorter,
            start=start,
            n_sims=n_sims,
            n_agents=n_agents,
            timesteps=timesteps,
            train=False,
            compute_fiv=compute_fiv
        )

        selection_priors, binnings, bias_priors = [], [], []

        for bias in (0, 1):
            for binning in self.bins:
                for selection in np.linspace(0, 1, n_sims):
                    selection_priors.append(selection)
                    binnings.append(binning)
                    bias_priors.append(bias)

        self.selection_priors = np.array(selection_priors)
        self.binnings = np.array(binnings)
        self.bias_priors = np.array(bias_priors)

        self.data = np.arange(len(self.binnings))


class DataLoader:
    def __init__(self, dataset: SimulationData, batch_size: int = 1, seed=None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = check_random_state(seed)
        self.seed = seed

    def __iter__(self):
        self.dataset.reset()
        if not self.dataset.train:
            self.rng = check_random_state(self.seed)

        bin_sizes = self.rng.choice(
            self.dataset.bins, size=len(self.dataset) // self.batch_size).tolist()

        while bin_sizes:
            n_bins = bin_sizes.pop()
            samples = []
            while len(samples) < self.batch_size:
                try:
                    samples.append(self.dataset.next(n_bins))
                except StopIteration:
                    break
            yield self.collate(samples)


    def collate(self, data):
        labels, selection, bins, outputs = zip(*data)
        return (
            torch.LongTensor(labels),
            torch.FloatTensor(selection),
            torch.LongTensor(bins),
            torch.stack(outputs),
        )
