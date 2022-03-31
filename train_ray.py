import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import numpy as np
import matplotlib.pyplot as plt
import LTCRL.utils as lru               # Utilities for training LTCs with pytorch
import LTCRL.models as models           # RayRLlib model implementations
from ray.rllib.examples.models.rnn_model import TorchRNNModel


ray.init(local_mode=True) # local mode = true : binds everything to a single process which enables easier debug

ModelCatalog.register_custom_model("RNN", models.NaiveRNN)

ModelCatalog.register_custom_model("LTC", models.LTC)

rnn_config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "Pendulum-v1",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 8,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "custom_model": "LTC",
        "custom_model_config": 
            {"sample_frequency": 9,
             "state_size": 12,
             "ode_unfolds": 5,
             "epsilon": 1e-8,
             },
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    }
}

# Instanciate the PPO trainer object
trainer = PPOTrainer(config=rnn_config)


# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
log = []
iterations = 1
for i in range(iterations):
    print("iteration : " +str(i),  ", ")
    log.append(trainer.train())
    print('len : ' + str(log[i]['episode_len_mean']))
    print('avg_rev : ' + str(np.array(log[i]['hist_stats']['episode_reward']).mean()))