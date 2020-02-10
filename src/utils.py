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


