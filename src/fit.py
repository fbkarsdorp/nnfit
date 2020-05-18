import arviz as az
import numpy as np
import scipy.stats as stats

from utils import suppress_stdout_stderr, StanModel_cache
from numba import jit


# @jit(nopython=True)
def frequency_increment_values(time, values, clip=False):
    if clip:
        values[values <= 0] = 0.001
        values[values >= 1] = 0.999
    n = len(time)
    Y = np.zeros(n - 1)
    for i in range(1, n):
        Y[i - 1] = (values[i] - values[i - 1]) / np.sqrt(
            2 * values[i - 1] * (1 - values[i - 1]) * (time[i] - time[i - 1]))
    return Y


def frequency_increment_test(time, values, clip=True):
    Y = frequency_increment_values(time, values, clip=clip)
    T, Tp = stats.ttest_1samp(Y, 0)
    W, Wp = stats.shapiro(Y)
    return {"T": T, "Tp": Tp, "W": W, "Wp": Wp}


def bayesian_frequency_increment_test(time, values, n_iter=10_000, ci=0.89):
    with open("ttest.stan") as f:
        sm = StanModel_cache(f.read(), model_name="ttest")
    Y = frequency_increment_values(time, values, clip=True)
    data = {"N": len(Y), "y": Y}
    fit = sm.sampling(data=data, chains=4, iter=n_iter)
    post = fit.extract(permuted=True)
    lci, uci = az.stats.hpd(post["mu"], credible_interval=ci)
    selection = (lci > 0) | (uci < 0)
    post = {
        "mu": np.mean(post["mu"]),
        "lci": lci,
        "uci": uci,
        "selection": selection,
        "fit": fit,
    }
    return post


def freq_demo():
    from simulation import wright_fisher

    for selection_strength in (0, 0.05, 0.2):
        d = wright_fisher(1000, 50, selection_strength) / 1000
        time = np.arange(1, d.shape[0] + 1)
        t, p = frequency_increment_test(time, d)
        print(f"Frequentatist t-test: strength={selection_strength}, t={t:.3f}, p={p:.3f}")

    
def bayesian_demo():
    from simulation import wright_fisher

    for selection_strength in (0, 0.05, 0.2):
        d = wright_fisher(1000, 50, selection_strength) / 1000
        time = np.arange(1, d.shape[0] + 1)
        with suppress_stdout_stderr():
            post = bayesian_frequency_increment_test(time, d)
        print(f"Bayesian t-test: strength={selection_strength}, mu={post['mu']:.3f}, "
              f"CI: [{post['lci']:.2f}, {post['uci']:.2f}]")


def demo():
    freq_demo()
    bayesian_demo()
