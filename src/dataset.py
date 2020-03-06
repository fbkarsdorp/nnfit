import concurrent

from typing import Tuple, Dict

import numpy as np
import scipy.stats as stats
import torch.utils.data

from utils import apply_binning, Distorter, check_random_state, loguniform
from simulation import wright_fisher
from fit import frequency_increment_values


class SimulationBatch:
    def __init__(
        self,
        distortion: Tuple[float, float] = None,
        variable_binning = False,
        start: float = 0.5,
        varying_start_value: bool = False,
        n_sims: int = 1000,
        n_agents: int = 1000,
        timesteps: int = 200,
        n_bins: int = None,
        seed: int = None,
        compute_fiv: bool = False,
    ) -> None:

        self.rng = check_random_state(seed)
        self.distortion = distortion
        if distortion is not None:
            self.distortion = Distorter(*distortion, seed=self.rng)
        if variable_binning and self.distortion is None:
            raise ValueError("Variable binning requires distortion prior")
        self.variable = variable_binning
        self.data = np.arange(n_sims)
        self.start = start
        self.varying_start_value = varying_start_value
        self.n_agents = n_agents
        self.timesteps = timesteps
        self.n_bins = n_bins
        self.compute_fiv = compute_fiv
        self.seed = seed
        self.n_samples = 0

        self.set_priors()

    def __len__(self):
        return self.data.shape[0]

    def set_priors(self) -> None:
        # Preset random values for reuse in validation
        n = len(self)
        self.selection_priors = loguniform(low=0.001, high=1, size=n, random_state=self.rng)
        self.bias_priors = self.rng.random(n)
        if self.varying_start_value:
            self.start = self.rng.uniform(0.001, 1, size=n)
        else:
            self.start = np.full(n, self.start)

    def _next(self):
        if self.n_samples < len(self.data):
            s = self.selection_priors[self.n_samples]
            if self.bias_priors[self.n_samples] < 0.5:
                s = 0

            j = wright_fisher(
                self.n_agents,
                self.timesteps,
                s,
                start=self.start[self.n_samples],
                random_state=self.rng,
            )

            j = apply_binning(
                j,
                self.n_bins,
                self.n_agents,
                variable=self.variable,
                random_state=self.rng,
            )

            if self.distortion is not None:
                j = self.distortion.distort(j)

            self.n_samples += 1

            biased = int(s != 0)

            if self.compute_fiv:
                j = frequency_increment_values(np.arange(1, len(j) + 1), j, clip=True)

            return biased, s, self.n_bins, torch.FloatTensor(j)

        raise StopIteration

    def next(self):
        samples = []
        while len(samples) < len(self):
            try:
                samples.append(self._next())
            except StopIteration:
                break
        labels, selection, bins, outputs = zip(*samples)
        return (
            torch.LongTensor(labels),
            torch.FloatTensor(selection),
            torch.LongTensor(bins),
            torch.stack(outputs),
        )


class DataLoader:
    def __init__(
        self,
        params: Dict,
        batch_size: int = 1,
        n_sims: int = 1000,
        distortion: Tuple[float, float] = None,
        seed: int = None,
        min_bin_length: int = 4,
        train: bool = True,
        n_workers: int = 1,
    ) -> None:
        
        self.params = params
        self.batch_size = batch_size
        self.rng = check_random_state(seed)
        self.seed = seed
        self.distortion = distortion
        self.n_sims = n_sims
        self.train = train
        self.n_workers = n_workers

        # sample bins following Karjus
        bins = {}
        maxlen = np.ceil(params["timesteps"] / min_bin_length)
        for i in range(1, int(maxlen + 1)):
            bins[i] = int(np.ceil(params["timesteps"] / i))
        bins = sorted(set(bins.values()))
        self.bins = np.array(bins)

        self.batch_seed_sequence = np.random.SeedSequence(seed)
        self.seed = seed

    def __iter__(self):
        if not self.train:
            self.batch_seed_sequence = np.random.SeedSequence(self.seed)
            self.rng = check_random_state(self.seed)

        bin_sizes = self.rng.choice(
            self.bins, size=self.n_sims // self.batch_size
        ).tolist()

        with concurrent.futures.ProcessPoolExecutor(self.n_workers) as executor:
            futures = [
                executor.submit(
                    SimulationBatch(
                        distortion=self.distortion,
                        variable_binning=self.params["variable_binning"],
                        start=self.params["start"],
                        varying_start_value=self.params["varying_start_value"],
                        n_sims=self.batch_size,
                        n_agents=self.params["n_agents"],
                        timesteps=self.params["timesteps"],
                        n_bins=bin_size,
                        seed=np.random.default_rng(rng),
                        compute_fiv=self.params["compute_fiv"],
                    ).next
                )
                for bin_size, rng in zip(
                    bin_sizes, self.batch_seed_sequence.spawn(len(bin_sizes))
                )
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    yield future.result()
                except Exception as exc:
                    print('generated an exception: %s' % (exc,))

# TODO: refactor
class TestSimulationData(SimulationBatch):
    def __init__(
        self,
        distortion=None,
        variable_binning=False,
        start=0.5,
        n_sims=1000,
        n_bins=25,
        n_agents=1000,
        timesteps=200,
        compute_fiv=False,
    ) -> None:

        super().__init__(
            distortion=distortion,
            start=start,
            n_sims=n_sims,
            n_agents=n_agents,
            timesteps=timesteps,
            train=False,
            compute_fiv=compute_fiv,
        )

        self.n_sims = n_sims
        self.init_start = start
        self.init_samples()

    def init_samples(self):
        self.n_samples = 0

        selection_priors, binnings, bias_priors, start_values = [], [], [], []
        selection_values = np.concatenate(
            ([0], np.exp(np.linspace(np.log(0.001), np.log(1), 25 - 1)))
        )

        for bias in (0, 1):
            for binning in self.bins:
                for selection in selection_values:
                    for rep in range(10):
                        selection_priors.append(selection)
                        binnings.append(binning)
                        start_values.append(self.init_start)
                        bias_priors.append(bias)

        self.selection_priors = np.array(selection_priors)
        self.binnings = np.array(binnings)
        self.bias_priors = np.array(bias_priors)
        self.start = np.array(start_values)

        self.data = np.arange(len(self.binnings))


class TestLoader(DataLoader):
    def __init__(self, dataset: TestSimulationData, batch_size: int = 1) -> None:
        super().__init__(dataset, batch_size)

    def __iter__(self):
        self.dataset.init_samples()
        while self.dataset.n_samples < len(self.dataset.data):
            samples = []
            while len(samples) < self.batch_size:
                try:
                    samples.append(self.dataset.next())
                except StopIteration:
                    break
            yield self.collate(samples)
