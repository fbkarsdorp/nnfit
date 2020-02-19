import argparse
import multiprocessing as mp
import tqdm

import numpy as np
import pandas as pd

from simulation import wright_fisher
from fit import frequency_increment_test
from utils import apply_binning, Distorter


def simulate(args):
    N, T, ss, n_bins, start = args
    data = wright_fisher(N, T, ss, start=start)
    data = apply_binning(data, n_bins, N)
    time = np.arange(1, data.shape[0] + 1)

    fit = frequency_increment_test(time, data)
    return {
        "start": start,
        "bin_size": n_bins,
        "selection_strength": ss,
        "t_statistic": fit["T"],
        "t_test_p_value": fit["Tp"],
        "shapiro_statistic": fit["W"],
        "shapiro_p_value": fit["Wp"],
        "selection": int(fit["Tp"] < 0.05),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000, help="Population size.")
    parser.add_argument("--T", type=int, default=200, help="Number of timesteps.")
    parser.add_argument("--sims", type=int, default=1000, help="Number of repetitions.")
    parser.add_argument("--bins", nargs="+", default=None, help="Bin sizes.")
    parser.add_argument(
        "--n_sel", type=int, default=200, help="Number of selection values."
    )
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers.")
    parser.add_argument(
        "--max_s", type=float, default=5, help="Max selection strength."
    )
    parser.add_argument(
        "--start", type=float, default=0.5, help="Relative start frequency."
    )
    parser.add_argument("output", type=str, help="Path to output file.")
    args = parser.parse_args()

    selection_values = np.concatenate(
        ([0], np.exp(np.linspace(np.log(0.001), np.log(args.max_s), args.n_sel - 1)))
    )

    with mp.Pool(args.n_workers) as pool:

        results = pool.map(
            simulate,
            (
                (args.N, args.T, ss, binning, args.start)
                for _ in range(args.sims)
                for binning in args.bins
                for ss in selection_values
            ),
        )

    results = pd.DataFrame(results)

    results.to_csv(args.output, index=False)
