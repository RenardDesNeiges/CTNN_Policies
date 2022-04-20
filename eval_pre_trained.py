import ray
from ray import tune
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import LTCRL.models as models

# register the rllib models
ModelCatalog.register_custom_model("RNN", models.NaiveRNN)
ModelCatalog.register_custom_model("LTC", models.LTC)

NUM_CPUS = 0 # changed based on PC / cluster
num_samples_each_worker = 0

eval_freq = 2


input_file = "/Users/renard/ray_results/MARWILTrainer_2022-04-18_20-23-30/MARWILTrainer_CartPole-v1_a5ec6_00000_0_2022-04-18_20-23-30/checkpoint_000095/checkpoint-95"

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
    "input": "~/ray_results/perceptron_baseline", # Expert policy data file
    "env": "CartPole-v1", # Environment
    # "model" : model,
    "framework": "torch", 
    # No need to calculate advantages (or do anything else with the
    # rewards).
    "beta": 0.0,
    # Advantages (calculated during postprocessing) not important for
    # behavioral cloning.
    "postprocess_inputs": False,
    # No reward estimation.
    "input_evaluation": [],
    
    # Parralelization stuff
    "num_workers": NUM_CPUS, # number of worker envs
    "train_batch_size": NUM_CPUS*num_samples_each_worker, 
    
    
    "evaluation_interval": eval_freq,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
        "steps" : 100,
        "episodes" : 1,
    },
}


# ray code
ray.init(local_mode=False)
# Instanciate the PPO trainer object
trainer = MARWILTrainer(config=config)

# trainer = PPOTrainer(config=config)
trainer.load_checkpoint(input_file)
trainer.evaluate()
print("Success!")


 