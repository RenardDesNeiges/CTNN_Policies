import ray
from ray import tune
from ray.rllib.agents.marwil import MARWILTrainer
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
}

# ray code
ray.init(local_mode=False)
# Instanciate the PPO trainer object
tune.run(MARWILTrainer, config=config)


