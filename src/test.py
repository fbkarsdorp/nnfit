import collections
import concurrent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn.cm as cm
import torch

from matplotlib.ticker import ScalarFormatter, FuncFormatter
from sklearn.metrics import accuracy_score, f1_score

from fit import frequency_increment_test, frequency_increment_values
from utils import apply_binning
from simulation import wright_fisher


def get_bins(timesteps, min_bin_length=4):
    bins = {}
    maxlen = np.ceil(timesteps / min_bin_length)
    for i in range(1, int(maxlen + 1)):
        bins[i] = int(np.ceil(timesteps / i))
    bins = np.array(sorted(set(bins.values())))
    return bins


def generate_test_samples(
    timesteps: int,
    n_agents: int,
    n_reps: int = 100,
    min_bin_length: int = 4,
    n_workers: int = 1,
    seed: int = None,
):

    selection_values = np.concatenate(
        ([0], np.exp(np.linspace(np.log(0.001), np.log(1), 200 - 1)))
    )

    samples = []
    for rep in range(n_reps):
        with concurrent.futures.ProcessPoolExecutor(n_workers) as executor:
            futures = {
                executor.submit(
                    wright_fisher, n_agents, timesteps, selection): selection
                for selection in selection_values
            }
            samples.append(
                [
                    (futures[future], future.result())
                    for future in concurrent.futures.as_completed(futures)
                ]
            )
    return samples


def _clean_up_nn_results(df):
    df["FP"] = (df["selection"] == 0) & (df["pred"] == True)
    df["FN"] = (df["selection"] > 0) & (df["pred"] != True)
    df["F"] = df["FN"] | df["FP"]
    return df


def _clean_up_fit_results(df):
    df["FP"] = (df["selection"] == 0) & (df["Tp"] < 0.05)
    df["FN"] = (df["selection"] > 0) & (df["Tp"] > 0.05)
    df["F"] = df["FN"] | df["FP"]
    df["pred"] = (df["Tp"] < 0.05).astype(int)
    df["y_true"] = (df["selection"] > 0).astype(int)
    df["normal"] = 1
    df.loc[df["Wp"] < 0.05, "normal"] = np.nan
    return df


def evaluate_results(df, prefix=""):
    df['selection_bin'] = pd.cut(
        df['selection'],
        [-0.001, 0.001, 0.005, 0.01, 0.1, 1],
        labels=["ß = 0", "0.001 ≤ ß ≤ 0.005", "0.005 ≤ ß ≤ 0.01", "0.01 ≤ ß ≤ 0.1", "0.1 ≤ ß ≤ 1"]
    )    

    return {
        # compute global scores (accuracy and f1)
        f"{prefix}accuracy": accuracy_score(*df[['y_true', 'pred']].values.T),
        f"{prefix}f1": f1_score(*df[['y_true', 'pred']].values.T),

        # compute false negative rate for max and min bin
        f"{prefix}FP_min_bin": df.loc[(df["selection"] == 0) & (df["bin"] == df["bin"].min()), "FP"].mean(),
        f"{prefix}FP_max_bin": df.loc[(df["selection"] == 0) & (df["bin"] == df["bin"].max()), "FP"].mean(),

        # compute false positive rate for max and min bin and low selection
        f"{prefix}FN_min_bin": df.loc[
            (df["selection"] > 0) &
            (df["selection"] < 0.005) &
            (df["bin"] == df["bin"].min()), "FP"
        ].mean(),
        f"{prefix}FN_max_bin": df.loc[
            (df["selection"] > 0) &
            (df["selection"] < 0.005) &
            (df["bin"] == df["bin"].max()), "FP"
        ].mean(),

        # Compute f1 scores for max and min bin with low selection
        f"{prefix}f1_min_bin": f1_score(*df.loc[
            (df["selection"] < 0.005) &
            (df["bin"] == df["bin"].min()), ["y_true", "pred"]].values.T),
        f"{prefix}f1_max_bin": f1_score(*df.loc[
            (df["selection"] < 0.005) &
            (df["bin"] == df["bin"].max()), ["y_true", "pred"]].values.T),
    }
    

def test_model(
        model, samples, timesteps, n_agents, min_bin_length=4, device="cuda", normalize_samples=False):

    bins = get_bins(timesteps, min_bin_length)

    nn_results, fit_results = [], []
    for batch in samples:
        data_dict = collections.defaultdict(list)
        for selection, data in batch:
            for bin_size in bins:
                binned_data = apply_binning(data, bin_size, n_agents)
                fit = frequency_increment_test(
                    np.arange(1, len(binned_data) + 1), np.array(binned_data)
                )
                fit.update({"selection": selection, "bin": bin_size})
                fit["pred"] = int(fit["Tp"] < 0.05)
                fit["y_true"] = int(selection > 0)
                fit_results.append(fit)

                if normalize_samples:
                    binned_data = np.array(binned_data)
                    binned_data = (binned_data - binned_data.mean()) / binned_data.std()
                data_dict[bin_size].append((selection, torch.FloatTensor(binned_data)))

        for bin_size, data in data_dict.items():
            selection, data = zip(*data)
            data = torch.stack(data).unsqueeze(1).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(data)).cpu().numpy().squeeze(1)
                preds = probs > 0.5
                for i, pred in enumerate(preds.tolist()):
                    nn_results.append(
                        {
                            "selection": selection[i],
                            "bin": bin_size,
                            "pred": int(pred),
                            "prob": probs[i],
                            "y_true": int(selection[i] > 0),
                        }
                    )
    
    nn_results = _clean_up_nn_results(pd.DataFrame(nn_results))
    fit_results = _clean_up_fit_results(pd.DataFrame(fit_results))
    return nn_results, fit_results


def _plot_heat(df, bins, ax=None, ylabel=None, title=None, mask=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    if mask is not None:
        df[mask] = np.nan

    selection_values = np.concatenate(
            ([0], np.exp(np.linspace(np.log(0.001), np.log(1), 200 - 1)))
        )
    
    img = ax.imshow(
        df, aspect="auto", cmap=cm.rocket, vmin=0, vmax=1)
    
    # correct x-ticks
    ticks = np.exp(np.linspace(np.log(0.001), np.log(1), 4))
    locs = np.where(np.isin(selection_values, ticks))[0]
    ax.set_xticks(locs)
    ax.set_xticklabels([f'{s:.13g}' for s in ticks])
    
    # correct y-ticks
    ax.set_yticks(np.arange(bins.shape[0]))
    ax.set_yticklabels(bins)
    
    # Set labels
    ax.set(xlabel="selection strength", ylabel=ylabel, title=title)
    return ax, img


def plot_parameter_sweep(fit_df, nn_df):
    fig, (fit_ax, fit_norm_ax, nn_ax) = plt.subplots(ncols=3, figsize=(18, 4))

    
    bins = np.array(sorted(fit_df["bin"].unique()))
    F = fit_df.groupby(["selection", "bin"])["FN"].mean().unstack().T
    ax, img = _plot_heat(F, bins, ax=fit_ax, title="(A) Frequency Increment Test", ylabel="number of bins")

    F = (fit_df.groupby(["selection", "bin"])["FN"].mean().unstack().T * 
         fit_df.groupby(["selection", "bin"])["normal"].mean().unstack().T)
    ax, img = _plot_heat(F, bins, ax=fit_norm_ax, title="(B) Frequency Increment Test")

    F = nn_df.groupby(["selection", "bin"])["FN"].mean().unstack().T
    ax, img = _plot_heat(F, bins, ax=nn_ax, title="(C) Time Series Classifier")

    fig.colorbar(img, ax=(fit_ax, fit_norm_ax, nn_ax))

    return fig


def mean_confidence_interval(data, confidence=0.89):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


def plot_bin_error(df, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    x = df.loc[(df['selection'] == 0), ['bin', 'FP']].astype(
        int).groupby('bin')['FP'].agg(mean=np.mean, ci=mean_confidence_interval).values
    mean = x[:, 0]
    ci = np.array(x[:, 1].tolist())
    bins = sorted(df["bin"].unique())
    ax.plot(bins, mean, '-x', label=label, lw=2)
    ax.fill_between(bins, ci[:, 0], ci[:, 1], alpha=0.3, color="grey")
    ax.set(xlabel="number of bins", ylabel="false positives rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    return ax


def plot_fp_scores(fit_df, nn_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_bin_error(fit_df, ax=ax, label="FIT")
    plot_bin_error(nn_df, ax=ax, label="classifier")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(frameon=False);
    return fig
