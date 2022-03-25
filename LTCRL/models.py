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



class LTC(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # TODO : process the relevant parameters here
        self.elapsed_time = 0.2 #kind of a random value, think about this 
        
        # TODO : declare the nn functional stuff here
    
    
    # TODO : pytorch sigmoid implementation 
    def _sigmoid(self, v_pre, mu, sigma):
        pass

    # TODO : FUSED step ODE solver
    def _ode_solver(self, inputs, state, elapsed_time):
        pass

    # TODO : input mapping function
    def _map_inputs(self, inputs):
        pass
    
    # TODO : output mapping function
    def _map_outputs(self, state):
        
        pass
        action = 0
        value = 0
        return action, value

    # TODO implement
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

