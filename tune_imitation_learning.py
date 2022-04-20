import ray
from ray import tune
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.models import ModelCatalog
import LTCRL.models as models

# register the rllib models
ModelCatalog.register_custom_model("RNN", models.NaiveRNN)
ModelCatalog.register_custom_model("LTC", models.LTC)

NUM_CPUS = 1 # changed based on PC / cluster
num_samples_each_worker = 2048 # int(32768 / NUM_CPUS)

eval_freq = 2

# input_directory = "~/ray_results/perceptron_baseline"
input_directory = "~/ray_results/RNN_baseline"

model = {
        "custom_model": "LTC",
        "custom_model_config": 
            {"sample_frequency": 10,
            "state_size": 8,
            "ode_unfolds": 5,
            "epsilon": 1e-8,
            },
        }

# environment config
config = {
    "input": input_directory, # Expert policy data file
    "env": "CartPole-v1", # Environment
    # "model" : model,
    "model": {
        "custom_model": "RNN",
        "max_seq_len": 50,
        # "custom_model_config": 
        #     {"sample_frequency": 10,
        #     "state_size": 8,
        #     "ode_unfolds": 5,
        #     "epsilon": 1e-8,
        #     },
        },
    "framework": "torch", 
    # No need to calculate advantages (or do anything else with the
    # rewards).
    "beta": 0.0,
    # Advantages (calculated during postprocessing) not important for
    # behavioral cloning.
    "postprocess_inputs": False,
    # No reward estimation.
    "_disable_execution_plan_api": False,
    # "replay_sequence_length": 30,
    "input_evaluation": [],
    
    # Parralelization stuff
    "num_workers": NUM_CPUS, # number of worker envs
    "train_batch_size": NUM_CPUS*num_samples_each_worker, 
    
    
    "evaluation_interval": eval_freq,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
    
    # === Replay Settings ===
    # The number of contiguous environment steps to replay at once. This may
    # be set to greater than 1 to support recurrent models.
    # "replay_sequence_length": 1,
}

# ray code
ray.init(local_mode=True)
# Instanciate the PPO trainer object
tune.run(MARWILTrainer, config=config, 
        checkpoint_freq=1,checkpoint_at_end=True)


