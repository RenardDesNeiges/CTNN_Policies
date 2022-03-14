# Learning Motor Policies with Time Continuous Neural Networks

Repository for my semester project at EPFL's Biorobotics Laboratory (BIOROB). The goal is to train time-continuous neural networks (more specifically LTCs) for strongly non-linear control tasks (ideally locomotion of quadruped robots) using reinforcement learning.

## Plan :
- Week 1 : getting started, picking a title **Done, on track**
- Week 2 : Understanding how LTCs work, supervised training, key properties **Done, on track**
- Week 3 : Start of the RL work (presumably in ray RLLib), RL training of RNNs for control tasks **Partially, not with RNNs**
- Week 4 : Implementing LTCs in ray, figuring out how to make BPTT work for RL (presumably using Proximal Policy Optimization) 
- Week 5 : Running examples in ray **Meeting Ijspeert + Gerstner**
- Week > 5 : end of the receding horizon, we will see later

## Possible Project Directions :
Open questions, this is quite an open ended project, this is where I store ideas where this might go :
- Proof Lyapunov stability of a closed loop system with LTCs under specific conditions
- Study in the effect of network topology and size on the results, which topology to pick to control a complex system
- Comparison with LSTMs and other RNNs
- Comparison of search based v.s. gradient based RL
- Get this to to work on a more complex environment (pybullet?)

## Software Installation and Organization : 

To install the right environment, use conda and install the following packages (TODO, make an install shell later):

```
conda env create --name torchNCP python=3.8
conda activate torchNCP
pip install torch keras-ncp gym "ray[rllib]" "gym[atari]" "gym[accept-rom-license]" atari_py matplotlib  pytorch-lightning
conda install pandas
conda install jupyter
```

The following files are relevant for the project:

```
.
├── LTCRL                       # utilities module for RL with LTCs
|   ├── __init__.py
|   └── utils.py
├── torch_NCP_example.ipynb     # supervised training examples
├── ray_RLLIB_example.ipynb     # Ray RL-Lib examples
└── data                        # test datasets
    ├── climate
    └── ozone
```
