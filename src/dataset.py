
import numpy as np
import scipy.stats as stats
import torch.utils.data
import torch.nn.functional as F

from typing import Tuple
from utils import apply_binning, Distorter, check_random_state
from simulation import wright_fisher


class SimulationData:
    def __init__(
        self,
        selection_prior: Tuple[float, float] = (1, 5),
        distortion: Distorter = None,
        variable_binning=False,
        start: float = 0.5,
        n_bins: int = 1,
        n_sims: int = 1000,
        n_agents: int = 1000,
        timesteps: int = 200,
        seed: int = None,
        train: bool = True,
        min_bin_length: int = 4
    ) -> None:

        self.rng = check_random_state(seed)
        self.selection_prior = stats.beta(selection_prior[0], selection_prior[1])
        self.selection_prior.random_state = self.rng
        self.distortion = distortion
        if variable_binning and self.distortion is None:
            raise ValueError("Variable binning requires distortion prior")
        self.variable = variable_binning
        self.data = np.arange(n_sims)
        self.start = start
        self.n_agents = n_agents
        self.timesteps = timesteps
        self.train = train
        self.seed = seed

        # sample bins following Karjus
        bins = {}
        maxlen = np.ceil(timesteps / min_bin_length)
        for i in range(1, int(maxlen + 1)):
            bins[i] = int(np.ceil(timesteps / i))
        bins = sorted(set(bins.values()))
        self.bins = np.array(bins)

        self.set_priors()

    def set_priors(self) -> None:
        # Preset random values for reuse in validation
        n = len(self.data)
        self.selection_priors = self.selection_prior.rvs(n)
        self.bias_priors = self.rng.rand(n)
        self.binnings = self.rng.choice(self.bins, size=n)

    def __iter__(self):
        self.n_samples = 0
        if not self.train:
            # Reinitialize Random Number Generator for consistent output
            self.rng = check_random_state(self.seed)
            # When using a distorter, reset it for consitent output in validation
            if self.distortion is not None:
                self.distortion.reset()
        if self.train:
            self.set_priors()
        return self

    def __next__(self):
        if self.n_samples < len(self.data):
            s = self.selection_priors[self.n_samples]
            if self.bias_priors[self.n_samples] < 0.5:
                s = 0

            j = wright_fisher(
                self.n_agents, self.timesteps, s, start=self.start, random_state=self.rng)

            # apply distortions
            distortions = None
            if self.distortion is not None:
                distortions = self.distortion.distort(j)
                j -= distortions

            # binning
            n_bins = self.binnings[self.n_samples]
            j = apply_binning(j, n_bins, self.n_agents, variable=self.variable,
                              distortions=distortions, random_state=self.rng)
            self.n_samples += 1

            biased = int(s != 0)

            return biased, s, n_bins, torch.FloatTensor(j)

        raise StopIteration


class TestSimulationData(SimulationData):
    def __init__(self, distorter=None, start=0.5, n_sims=1000, n_bins=25, n_agents=1000, timesteps=200):
        super().__init__(
            distortion=distorter
            start=start,
            n_sims=n_sims,
            n_agents=n_agents,
            n_bins=n_bins,
            timesteps=timesteps,
            train=False,
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
    def __init__(self, dataset: SimulationData, batch_size: int = 1) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        dataset = iter(self.dataset)
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) == self.batch_size:
                yield self.collate(samples, dataset.timesteps)
                samples = []
        if samples:
            yield self.collate(samples, dataset.timesteps)

    def collate(self, data, length):
        labels, selection, bins, outputs = zip(*data)
        padded_outputs = []
        for i, output in enumerate(outputs):
            padded_outputs.append(
                F.pad(output, (0, length - output.size(0)), "constant", 0)
            )
        return (
            torch.LongTensor(labels),
            torch.FloatTensor(selection),
            torch.LongTensor(bins),
            torch.stack(padded_outputs),
        )
