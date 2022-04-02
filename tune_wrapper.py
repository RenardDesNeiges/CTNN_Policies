import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import LTCRL.models as models

# register the rllib models
ModelCatalog.register_custom_model("RNN", models.NaiveRNN)
ModelCatalog.register_custom_model("LTC", models.LTC)


model = {
        "custom_model": "RNN",
        # "custom_model_config": 
        #     {"sample_frequency": 9,
        #     "state_size": 12,
        #     "ode_unfolds": 5,
        #     "epsilon": 1e-8,
        #     },
        }

# environment config
config = {
    "env": "Cartpole-v1", # Environment
    "num_workers": 4, # number of worker envs
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