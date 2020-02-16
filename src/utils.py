import numbers
import pickle

from hashlib import md5

import os
import numpy as np
import scipy.stats as stats
import pystan


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState" " instance" % seed)


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


def apply_binning(counts, n_bins, n_agents, variable=False, random_state=None):
    rng = check_random_state(random_state)
    pop_size = np.full(counts.shape[0], n_agents)    

    if not variable:
        bins = np.array_split(np.arange(len(counts)), n_bins)
        return [(counts[b].sum() / pop_size[b].sum()) for b in bins]

    # assume variable
    # TODO: clean up
    output = np.zeros(sum(pop_size))
    cur = 0
    for count, c_pop_size in zip(counts.astype(int).tolist(), pop_size.tolist()):
        output[cur : cur + c_pop_size] = rng.permutation(
            [1] * count + [0] * (c_pop_size - count)
        )
        cur += c_pop_size
    output = np.array([sum(b) / len(b) for b in np.array_split(output, n_bins)])

    return output


class Distorter:
    def __init__(self, loc=0, sd=0.2, seed=None):
        self.prior = stats.norm(loc, sd)
        self.rng = check_random_state(seed)
        self.seed = seed

    def _sample(self, n):
        return self.prior.rvs(n, random_state=self.rng)

    def distort(self, values):
        values = np.array(values)
        distortion = self._sample(values.shape[0])
        return np.clip(values - distortion, 0, 1)

    def reset(self):
        loc, sd = self.prior.args
        self.prior = stats.norm(loc, sd)
        self.rng = check_random_state(self.seed)


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode("ascii")).hexdigest()
    if model_name is None:
        cache_fn = "cached-model-{}.pkl".format(code_hash)
    else:
        cache_fn = "cached-{}-{}.pkl".format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, "rb"))
    except:
        sm = pystan.StanModel(model_code=model_code, **kwargs)
        with open(cache_fn, "wb") as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
