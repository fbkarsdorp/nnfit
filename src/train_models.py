from train import run_experiment, get_arguments


CONFIG = {
    "n_sims": 5000,
    "batch_size": 200,
    "val_size": 1,
    "start": 0.5,
    "n_epochs": 1000,
    "selection_params": (1, 5),
    "learning_rate": 1e-5,
    "cuda": True,
    "learning_rate_patience": 5,
    "early_stop_patience": 10,
}


def make_path(n_agents, varying_start, timesteps, distortion, fiv, model, lr):
    start = "vstart" if varying_start else "fstart"
    fiv = "_fiv" if fiv else ""
    distortion = "_distortion" if distortion else ""
    return f'{n_agents}_{start}_{timesteps}{distortion}{fiv}_{model}_{lr}'


N_AGENTS = 5000, 10000
VARYING_START_VALUE = True,
LEARNING_RATE = 0.001, 1e-5, 6e-5
TIMESTEPS = 200, 
DISTORTION = True, False
COMPUTE_FIV = True, False
MODELS = "RESNET", "FCN"

for n_agents in N_AGENTS:
    for varying_start_value in VARYING_START_VALUE:
        for timesteps in TIMESTEPS:
            for distortion in DISTORTION:
                for fiv in COMPUTE_FIV:
                    for model in MODELS:
                        for learning_rate in LEARNING_RATE:
                            args = get_arguments()

                            # update settings
                            args.outfile = make_path(
                                n_agents, varying_start_value, timesteps, distortion, fiv, model, learning_rate)
                            args.n_agents = n_agents
                            args.varying_start_value = varying_start_value
                            args.timesteps = timesteps
                            args.distortion = distortion
                            args.compute_frequency_increment_values = fiv
                            args.model = model
                            args.learning_rate = learning_rate

                            dict_args = vars(args)
                            dict_args.update(CONFIG)

                            run_experiment(args)
