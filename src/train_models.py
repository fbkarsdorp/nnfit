import os

import pandas as pd

from train import run_experiment, get_arguments
from test import generate_test_samples, test_fit
from test import plot_parameter_sweep, plot_fp_scores


CONFIG = {
    "n_sims": 50000,
    "val_size": 1,
    "n_epochs": 1000,
    "cuda": True,
    "learning_rate_patience": 5,
    "early_stop_patience": 10,
    "n_workers": 40,
    "compute_frequency_increment_values": False,
    "test": True,
    "seed": 1983
}

result_file = "../results/test_results.csv"

for n_agents in (1000, 5000, 10000):
    for timesteps in (200, 500, 1000):
        print(f"Generating test samples for N={n_agents} and T={timesteps}.")
        test_samples = generate_test_samples(timesteps, n_agents, 1000, n_workers=10)
        print("Testing FIT preformance...")
        fit_results, fit_scores = test_fit(test_samples, timesteps, n_agents)
        print("Starting model training...")
        for batch_size in (100, 200, 500):
            for model in ("RESNET", "FCN", "INCEPTION"):
                for learning_rate in (0.001, 1e-5, 6e-5):
                    for normalize_samples in (False, True):
                        args = get_arguments()

                        # update settings
                        args.n_agents = n_agents
                        args.varying_start_value = True
                        args.timesteps = timesteps
                        args.compute_frequency_increment_values = False
                        args.model = model
                        args.batch_size = batch_size
                        args.learning_rate = learning_rate
                        args.normalize_samples = normalize_samples

                        dict_args = vars(args)
                        dict_args.update(CONFIG)

                        args.test_samples = test_samples

                        run_id, nn_results, nn_scores = run_experiment(args)
                        
                        plot_parameter_sweep(fit_results, nn_results).savefig(
                            f"../results/{run_id}_params_sweep.png", dpi=300)
                        plot_fp_scores(fit_results, nn_results).savefig(
                            f"../results/{run_id}_fp_scores.png", dpi=300)
                        
                        dict_args.update(nn_scores)
                        dict_args.update(fit_scores)
                        dict_args["runid"] = run_id


                        df = pd.DataFrame([{k: v for k, v in dict_args.items() if k != "test_samples"}])
                        if os.path.exists(result_file):
                            df = pd.concat((pd.read_csv(result_file), df))
                        df.to_csv(result_file, index=False)
