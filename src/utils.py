from hashlib import md5
import pickle

import os
import numpy as np
import scipy.stats as stats
import pystan


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

    def distort(self, values):
        loss_prior = self.prior.rvs(values.shape[0])
        return self.rnd.binomial(values.astype(int), loss_prior)


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
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

