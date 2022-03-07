# Learning Motor Policies with Time Continuous Neural Networks

Repository for my semester project at EPFL's Biorobotics Laboratory (BIOROB). The goal is to train time-continuous neural networks (more specifically LTCs) for strongly non-linear control tasks (ideally locomotion of quadruped robots) using reinforcement learning.

## Plan :
- Week 1 : getting started, picking a title
- Week 2 : Understanding how LTCs work, supervised training, key properties
- Week 3 : Start of the RL work (presumably in ray RLLib), RL training of RNNs for control tasks
- Week 4 : Implementing LTCs in ray, figuring out how to make BPTT work for RL (presumably using a PPO alg) **Meeting Ijspeert + Gerstner**
- Week 5 : Running examples in ray
- Week > 5 : end of the receding horizon, we will see later

## Software Logistics : 

To install the right environment, use conda and run the following command :

```
conda env create --name torchNCP --file=environment.yml
```

The following files are relevant for the project:

```
├── LTCRL                       # utilities module for RL with LTCs
|   ├── __init__.py
|   └── utils.py
├── torch_NCP_example.ipynb     # supervised training examples
├── ray_RLLIB_example.ipynb     # Ray RL-Lib examples
└── data                        # test datasets
    ├── climate
    └── ozone
```

