
import numpy as np
import scipy.stats as stats
import torch.utils.data
import torch.nn.functional as F

from typing import Tuple


def apply_binning(counts, n_bins, n_agents, rnd=None, variable=False, distortions=None):
    # produce relative frequency per bin
    if distortions is None:
        distortions = np.zeros_like(counts)
    pop_size = n_agents - distortions

    if not variable:
        bins = np.array_split(np.arange(len(counts)), n_bins)
        return [counts[b].sum() / pop_size[b].sum() for b in bins]

    # assume variable
    output = np.zeros(sum(pop_size))
    cur = 0
    for count, c_pop_size in zip(counts.astype(int).tolist(), pop_size.tolist()):
        output[cur: cur+c_pop_size] = (rnd or np.random).permutation(
            [1] * count + [0] * (c_pop_size - count))
        cur += c_pop_size
    output = np.array([sum(b) / len(b) for b in np.array_split(output, n_bins)])

    return output


class SimulationData:
    def __init__(
        self,
        selection_prior: Tuple[float, float] = (1, 5),
        distortion_prior: Tuple[float, float] = None,
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

        self.rnd = np.random.RandomState(seed)
        self.selection_prior = stats.beta(selection_prior[0], selection_prior[1])
        self.selection_prior.random_state = self.rnd
        self.distortion_prior = distortion_prior
        if self.distortion_prior is not None:
            self.distortion_prior = stats.beta(distortion_prior[0], distortion_prior[1])
            self.distortion_prior.random_state = self.rnd
        if variable_binning and not self.distortion_prior:
            raise ValueError("Variable binning requires distortion prior")
        self.variable = variable_binning
        self.data = np.arange(n_sims)
        self.start = int(start * n_agents)
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

    def set_priors(self):
        # Preset random values for reuse in validation
        n = len(self.data)
        self.selection_priors = self.selection_prior.rvs(n)
        self.distortion_priors = None
        if self.distortion_prior is not None:
            self.distortion_priors = self.distortion_prior.rvs(n)
        self.bias_priors = self.rnd.rand(n)
        self.binnings = self.rnd.choice(self.bins, size=n)

    def __iter__(self):
        self.n_samples = 0
        if not self.train:
            self.rnd = np.random.RandomState(self.seed)
        if self.train:
            self.set_priors()
        return self

    def __next__(self):
        if self.n_samples < len(self.data):
            s = self.selection_priors[self.n_samples]
            if self.bias_priors[self.n_samples] < 0.5:
                s = 0

            j = np.zeros(self.timesteps)
            j[0] = self.start
            for i in range(1, self.timesteps):
                p_star = j[i - 1] * (1 + s) / (j[i - 1] * s + self.n_agents)
                j[i] = self.rnd.binomial(self.n_agents, min(p_star, 1))

            # apply distortions
            distortions = None
            if self.distortion_prior:
                ps = self.distortion_priors[len(j)]
                distortions = self.rnd.binomial(j.astype(int), ps)
                j -= distortions

            # binning
            n_bins = self.binnings[self.n_samples]
            j = apply_binning(j, n_bins, self.n_agents, rnd=self.rnd,
                              variable=self.variable, distortions=distortions)
            self.n_samples += 1

            biased = int(s != 0)

            return biased, s, n_bins, torch.FloatTensor(j)

        raise StopIteration


class TestSimulationData(SimulationData):
    def __init__(self, start=0.5, n_sims=1000, n_bins=25, n_agents=1000, timesteps=200):
        super().__init__(
            start=start,
            n_sims=n_sims,
            n_agents=n_agents,
            n_bins=n_bins,
            timesteps=timesteps,
            train=False,
        )

        self.selection_priors = []
        self.binnings = []
        self.bias_priors = []

        for bias in (0, 1):
            for binning in self.bins:
                for selection in np.linspace(0, 1, n_sims):
                    self.selection_priors.append(selection)
                    self.binnings.append(binning)
                    self.bias_priors.append(bias)

        self.selection_priors = np.array(self.selection_priors)
        self.binnings = np.array(self.binnings)
        self.bias_priors = np.array(self.bias_priors)

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
