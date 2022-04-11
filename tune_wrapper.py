import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import LTCRL.models as models

# register the rllib models
ModelCatalog.register_custom_model("RNN", models.NaiveRNN)
ModelCatalog.register_custom_model("LTC", models.LTC)

NUM_CPUS = 8 # changed based on PC / cluster
num_samples_each_worker = int(16384 / NUM_CPUS)


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
    "num_envs_per_worker": 1, 
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    "lr": 1e-4,
    "env": "CartPole-v1", # Environment
    "num_workers": NUM_CPUS, # number of worker envs
    "train_batch_size": NUM_CPUS*num_samples_each_worker, 
    "num_sgd_iter": 10,
    "rollout_fragment_length": num_samples_each_worker,
    
    "vf_loss_coeff": 0.5,
    "lambda":0.95, 
    "grad_clip": 0.5, 
    #"use_kl_loss": False,
    # "kl_coeff": 0.0, 
    # "kl_target": 0.01,
    "entropy_coeff": 0.0,
    "observation_filter": "MeanStdFilter", # normalize observation space (very important)
    "clip_actions": True,
    "num_sgd_iter": 10,
    "clip_param": 0.2,
    "vf_clip_param": 40, # may need to increase depending on your reward scaling
    "sgd_minibatch_size": 256,
    "framework": "torch", # we run pytorch, not tensorflow
    "model": model,
    "evaluation_interval": 20,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    }
}

# ray code
ray.init(local_mode=False)
# Instanciate the PPO trainer object
tune.run(PPOTrainer, config=config)


