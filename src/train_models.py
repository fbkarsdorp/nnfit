from train import run_experiment, get_arguments


CONFIG = {
    "n_sims": 1000,
    "batch_size": 100,
    "val_size": 1,
    "start": 0.5,
    "model": "RESNET",
    "n_epochs": 100,
    "selection_params": (1, 5),
    "learning_rate": 0.01,
    "cuda": True,
    "compute_frequency_increment_values": True
}


def make_path(n_agents, varying_start, timesteps, distortion, fiv):
    start = "vstart" if varying_start else "fstart"
    fiv = "_fiv" if fiv else ""
    distortion = "_distortion" if distortion else ""
    return f'{n_agents}_{start}_{timesteps}{distortion}{fiv}'


N_AGENTS = list(range(1000, 11000, 1000))
VARYING_START_VALUE = True, False
TIMESTEPS = 4, 10, 20, 50, 100, 200, 500
DISTORTION = True, False

for n_agents in N_AGENTS:
    for varying_start_value in VARYING_START_VALUE:
        for timesteps in TIMESTEPS:
            for distortion in DISTORTION:
                args = get_arguments()

                # update settings
                args.outfile = make_path(n_agents, varying_start_value, timesteps, distortion)
                args.n_agents = n_agents
                args.varying_start_value = varying_start_value
                args.timesteps = timesteps
                args.distortion = distortion

                dict_args = vars(args)
                dict_args.update(CONFIG)
                
                run_experiment(args)
