import numpy as np
import tqdm

from utils import counters2array


class EvoModel:
    def __init__(self, N, mu=0.001, b=0, burn_in=1000, timesteps=10, seed=None, verbose=False):
        self.N = N
        self.mu = mu
        self.b = b
        self.burn_in = burn_in
        self.timesteps = timesteps
        self.rnd = np.random.RandomState(seed)
        self.verbose = verbose
        self.n_traits = N

        self.population = np.arange(self.N)
        self.trait_fd = []

    def run(self):
        for timestep in tqdm.trange(self.burn_in, disable=not self.verbose):
            self._sample(timestep)
        for timestep in tqdm.trange(self.timesteps, disable=not self.verbose):
            self._sample(timestep + self.burn_in)
            self.trait_fd.append(collections.Counter(self.population))
        self.trait_fd = counters2array(self.trait_fd)
        return self

    def _sample(self, timestep):
        # Compute frequency distribution over traits
        traits, counts = np.unique(self.population, return_counts=True)
        counts = (counts ** (1 - self.b))
        # Randomly assign new traits to population
        self.population = self.rnd.choice(
            traits, self.N, replace=True, p=counts / counts.sum())
        # Assign innovations to innovating individuals
        innovating_population = self.rnd.rand(self.N) < self.mu
        n_innovations = innovating_population.sum()
        self.population[innovating_population] = np.arange(
            self.n_traits, self.n_traits + n_innovations)
        self.n_traits += n_innovations
        

def wright_fisher(N, T, selection_strength, start=0.5, seed=None):
    rnd = np.random.RandomState(seed)
    series = np.zeros(T)
    series[0] = int(start * N)
    for i in range(1, T):
        p_star = (
            series[i - 1]
            * (1 + selection_strength)
            / (series[i - 1] * selection_strength + N)
        )
        series[i] = rnd.binomial(N, min(p_star, 1))
    return series
