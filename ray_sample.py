

ray.init()
# Neural Network parameters (may need to change depending on obs/action space sizes)
model = {"fcnet_hiddens": [512, 256], "fcnet_activation": "tanh"} 

### PPO 
config = ppo.DEFAULT_CONFIG.copy()
NUM_CPUS = 16 # changed based on PC / cluster
num_samples_each_worker = int(4096 / NUM_CPUS)

# some of these may need to be tuned for your environment, but should be a good starting point
config_PPO={"env": DummyEnv, 
            "num_gpus": 0,
            "num_workers": NUM_CPUS,
            "num_envs_per_worker": 1, 
            "lr": 1e-4,
            "monitor": True,
            "model": model,
            "train_batch_size": NUM_CPUS*num_samples_each_worker, 
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,
            "rollout_fragment_length": num_samples_each_worker,
            "clip_param": 0.2,
            "vf_clip_param": 1, # may need to increase depending on your reward scaling
            "vf_loss_coeff": 0.5,
            "lambda":0.95, 
            "grad_clip": 0.5, 
            #"use_kl_loss": False,
            # "kl_coeff": 0.0, 
            # "kl_target": 0.01,
            "entropy_coeff": 0.0,
            "observation_filter": "MeanStdFilter", # normalize observation space (very important)
            "clip_actions": True,
            "vf_share_layers": False, 
            "normalize_actions": True,
            "preprocessor_pref": "rllib", 
            "batch_mode": "truncate_episodes", 
            "framework": "tf", # either torch or tensorflow are fine
            "metrics_smoothing_episodes": 1,
            "no_done_at_end": False,
            "shuffle_buffer_size": NUM_CPUS*num_samples_each_worker, 

        }
config.update(config_PPO)
for k,v in config.items():
    print(k,v)

tune.run("PPO",config=config, 
                local_dir=SAVE_RESULTS_PATH,
                checkpoint_freq=20, 
                verbose=2,
                stop= {"timesteps_total": 1000000000},
                )

ray.shutdown()