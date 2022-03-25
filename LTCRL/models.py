import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class NaiveRNN(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 fc_size=64,
                 rnn_state_size=64):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.rnn_state_size = rnn_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.rnn = nn.RNN(
            self.fc_size, self.rnn_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.rnn_state_size, num_outputs)
        self.value_branch = nn.Linear(self.rnn_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = self.fc1.weight.new(1, self.rnn_state_size).zero_().squeeze(0)
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """
            Forward pass throught the RNN
        """

        state = torch.stack(state,1)
        
        x = nn.functional.relu(self.fc1(inputs))
        self._features, h = self.rnn( x, torch.unsqueeze(state, 0))
        action_out = self.action_branch(self._features)
        
        lh = list(h.squeeze(0).swapaxes(0,1))
        
        return action_out, lh


"""
    Ray RLLlib class implementing a Liquid Time Constant neural network, the style of the implemenation is really similar to the one found in the implementation of the NCP paper (https://github.com/mlech26l/keras-ncp) but it is build for RL with a action branch/value branch arch for actor critic methods
"""
class LTC(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        # TODO : process the relevant parameters here
        
        # parsing model kwargs : 
        self.elapsed_time = 0.2 #kind of a random value, think about this 
        if 'sample_frequency' in kwargs:
            self.elapsed_time = 1/kwargs['sample_frequency']
        
        # TODO : declare the nn functional stuff here
    
    # Utility function to add weights to the model
    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    
    # TODO : parameter allocation routine here
    def _allocate_parameters(self):
        
        print("alloc!")
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(self._wiring.erev_initializer()),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        self._params["sparsity_mask"] = torch.Tensor(
            np.abs(self._wiring.adjacency_matrix)
        )
        self._params["sensory_sparsity_mask"] = torch.Tensor(
            np.abs(self._wiring.sensory_adjacency_matrix)
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,)),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,)),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )
    
    # TODO : pytorch sigmoid implementation 
    def _sigmoid(self, v_pre, mu, sigma):
        pass

    # TODO : FUSED step ODE solver
    def _ode_solver(self, inputs, state, elapsed_time):
        pass

    # TODO : input mapping function
    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"] + self._params["input_b"]
    
    # TODO : output mapping function (should return an action and a value branch)
    def _map_outputs(self, state):
        
        pass
        action = 0
        value = 0
        return action, value

    # TODO implement the init states function (should return a tensor of hidden states size)
    @override(ModelV2)
    def get_initial_state(self):
        pass

    # TODO think of a relevant value function
    @override(ModelV2)
    def value_function(self):
        pass

    # TODO implement the LTC forward pass
    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """
            Forward pass throught the RNN
        """
        
        inputs = self._map_inputs(inputs) # input mapping
        next_state = self._ode_solver(inputs, state, self.elapsed_time) # compute the ODE step
        action, _ = self._map_outputs(next_state) # compute the ODE step
        
        
        
        return outputs, next_state

