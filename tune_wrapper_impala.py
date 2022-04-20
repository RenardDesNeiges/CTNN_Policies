import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.models import ModelCatalog
import LTCRL.models as models

# register the rllib models
ModelCatalog.register_custom_model("RNN", models.NaiveRNN)
ModelCatalog.register_custom_model("LTC", models.LTC)

NUM_CPUS = 8 # changed based on PC / cluster
num_samples_each_worker = int(4096 / NUM_CPUS)


model = {
    
        "custom_model": "LTC",
        "custom_model_config": 
            {"sample_frequency": 5,
            "state_size": 8,
            "ode_unfolds": 3,
            "epsilon": 1e-8,
            },
        }

# environment config
config = {
    "model": model,
    "num_envs_per_worker": 1, 
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "env": "CartPole-v1", # Environment
    "num_workers": NUM_CPUS, # number of worker envs
    
    "num_gpus": 0,
    "train_batch_size": NUM_CPUS*num_samples_each_worker, 
    
    
    
    # V-trace params (see vtrace_tf/torch.py).
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    # If True, drop the last timestep for the vtrace calculations, such that
    # all data goes into the calculations as [B x T-1] (+ the bootstrap value).
    # This is the default and legacy RLlib behavior, however, could potentially
    # have a destabilizing effect on learning, especially in sparse reward
    # or reward-at-goal environments.
    # False for not dropping the last timestep.
    "vtrace_drop_last_ts": True,
    "min_time_s_per_reporting": 10,
    
    
    "num_sgd_iter": 10,
    "rollout_fragment_length": num_samples_each_worker,

    # Learning params.
    "grad_clip": 40.0,
    # Either "adam" or "rmsprop".
    "opt_type": "adam",
    "lr": 0.0005,
    "lr_schedule": None,
    # `opt_type=rmsprop` settings.
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # Balancing the three losses.
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,
    # Set this to true to have two separate optimizers optimize the policy-
    # and value networks.
    "_separate_vf_optimizer": False,
    # If _separate_vf_optimizer is True, define separate learning rate
    # for the value network.
    "_lr_vf": 0.0005,

    "observation_filter": "MeanStdFilter", # normalize observation space (very important)

    "framework": "torch", # we run pytorch, not tensorflow

    "evaluation_num_workers": 0,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    }
}

# ray code
ray.init(local_mode=False)
# Instanciate the PPO trainer object
tune.run(ImpalaTrainer, config=config)


