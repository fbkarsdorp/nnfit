import numpy as np
import scipy.stats as stats


class Prior:
    def __init__(self, **priors):
        self.priors = priors

    def rvs(self):
        return {p: d.rvs() for p, d in self.priors.items()}


def counters2array(counters):
    max_value = max(max(counter.keys()) for counter in counters)
    array = np.zeros((len(counters), max_value + 1))
    for i, counter in enumerate(counters):
        for j, value in counter.items():
            array[i, j] = value
    return array


class Distorter:
    def __init__(self, prior=None, seed=None):
        if prior is None:
            prior = stats.beta(0.5, 1.5)
        self.prior = prior
        self.rnd = np.random.RandomState(seed)

    def distort(self, values, time=None):
        if time is None:
            time = np.arange(1, values.shape[0] + 1)

        loss_prior = self.prior.rvs(values.shape[0])
        values = values - self.rnd.binomial(values.astype(int), loss_prior)

        return values[values > 0], time[values > 0]
