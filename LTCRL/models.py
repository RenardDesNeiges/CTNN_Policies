import numpy as np

import kerasncp as kncp                 # Keras LTC implementation (to use the Wiring Class)

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

""" 
    Reference implementation of a simple Recurrent Neural Network
"""
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
        val = torch.reshape(self.value_branch(self._features), [-1])
        #print("value shape = {}".format(val.shape))
        return val

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
        #print("inputs : {}, next state : {}, len(lh) : {}, lh shape : {} , actions : {}".format(inputs.shape, h.shape, len(lh), lh[0].shape, action_out.shape))
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

        
        # initialization range for random initial parameters
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        
        # parsing model kwargs : 
        self._parse_kwargs(kwargs)
        self._obs_size = get_preprocessor(obs_space)(obs_space).size
        self._action_size = num_outputs
        self._wiring = kncp.wirings.FullyConnected(self.state_size,self._action_size) # build the LTC wiring object

        # we need to build the wiring before using it
        self._wiring.build(self._obs_size)
        if not self._wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        
        self._allocate_parameters()
        
        self.value_branch = nn.Linear(self.state_size, 1)

        pass
        # TODO : declare the nn functional stuff here
    
    # parses a single keyword argument
    def parse_one_kwarg(self, arg, kwargs, default_value):
        if arg in kwargs:
            return kwargs['sample_frequency']
        return default_value
    
    # TODO : process the relevant parameters here
    def _parse_kwargs(self, kwargs):
        
        # TODO : alocate missing ODE parameters
        self.elapsed_time = 1/self.parse_one_kwarg('sample_frequency',kwargs,5)
        self.state_size = self.parse_one_kwarg('state_size',kwargs, 10)
        self._ode_unfolds = self.parse_one_kwarg('ode_unfolds',kwargs, 5)
        self._epsilon = self.parse_one_kwarg('epsilon',kwargs, 1e-8)
    
    # Utility function to add weights to the model
    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param
    
    # initializes the parameters in a given range
    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    
    # parameter allocation routine here
    def _allocate_parameters(self):
        
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
            init_value=torch.Tensor(self._wiring.erev_initializer()), #TODO : implement that
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self._obs_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self._obs_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self._obs_size, self.state_size), "sensory_w"
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

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self._obs_size,)),
        )

        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self._obs_size,)),
        )

        self._params["output_w"] = self.add_weight(
            name="output_w",
            init_value=torch.ones((self._action_size,)),
        )
        self._params["output_b"] = self.add_weight(
            name="output_b",
            init_value=torch.zeros((self._action_size,)),
        )

        self._params["value_w"] = self.add_weight(
            name="output_w",
            init_value=torch.ones((self._action_size,)),
        )
        self._params["output_b"] = self.add_weight(
            name="output_b",
            init_value=torch.zeros((self._action_size,)),
        )
        
        # TODO : action branch params (a and b)

    # pytorch sigmoid implementation 
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    # FUSED step ODE solver
    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state
        # sensory_inputs = 
        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation *= self._params["sensory_sparsity_mask"].to(
            sensory_w_activation.device
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation *= self._params["sparsity_mask"].to(w_activation.device)

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self._params["gleak"] * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self._params["gleak"] + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    # input mapping function
    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"] + self._params["input_b"]
        return inputs
    
    # output mapping function (returns to the action branch)
    def _map_outputs(self, state):
        
        action = state
        if self._action_size < self.state_size:
            action = action[:, 0 : self._action_size]  # slice

        action = action * self._params["output_w"] + self._params["output_b"] # affine mapping

        return action

    # init network states
    @override(ModelV2)
    def get_initial_state(self):
        
        # TODO : find some way to set the device
        h = torch.zeros(
            (1, self.state_size)).squeeze(0)

        return h

    # value function
    @override(ModelV2)
    def value_function(self):
        # TODO : fix the value branch
    
        assert self._features is not None, "must call forward() first"
        val = torch.reshape(self.value_branch(self._features), [-1])
        #print("value shape = {}".format(val.shape))
        return val
        # return torch.sum(self.value_branch(self._features),dim=1)
        

    # TODO implement the LTC forward pass
    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """
            Forward pass throught the LTC Cell, constructs time sequences 
            that can be trained with through backpropagation through time [BPTT]
            ----------------------------------------------------------------
            arguments : 
                inputs      <-  (BATCH_SIZE x SEQ_LEN x OBS_SIZE) torch tensor
                                inputs in time along the rnn sequence
                state       <-  STATE_SIZE list of (BATCH_SIZE) torch tensors 
                                gives state at the begining of the RNN sequences
                                (for some obscure rllib reason)
                seq_lens    <-  (BATCH_SIZE) torch tensor containing
                                sequence lens for each seq in the batch
            ----------------------------------------------------------------
            outputs :
                action      <-  (BATCH_SIZE x SEQ_LEN x ACTION_SIZE) torch tensor
                                outputs in time along the rnn sequence
                                this is then processe by the action distribution
                lh          <-  STATE_SIZE list of (BATCH_SIZE) torch tensors 
                                state at the end of the RNN sequences
                                (for some obscure rllib reason)
        """
        
        
        device = inputs.device
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # states should be :  states * (seq)
        actions = []
        self._features = []
        next_state = torch.stack(state,1)
        for i in range(seq_len):
            I_t = self._map_inputs(torch.squeeze(inputs[:,i,:],1)) # input mapping
            next_state = self._ode_solver(I_t, next_state, self.elapsed_time) # compute the ODE step 
            new_action = self._map_outputs(next_state) # compute the ODE step
            actions.append(new_action)
            self._features.append(next_state)
        
        action = torch.stack(actions,1)
        self._features = torch.stack(self._features,1)
        lh = list(next_state.swapaxes(0,1))
        #print("inputs : {}, next state : {}, len(lh) : {}, lh shape : {}, actions : {}".format(inputs.shape, next_state.shape, len(lh), lh[0].shape, action.shape))
                
        return action, lh
        # lh should be :  states * (seq)
        # actions should be : 

    # def forward(self, x): # just recursively constructs a time-sequence with outputs that cover the entire sequence (which is required for BPTT)
    #     # device = x.device
    #     # batch_size = x.size(0)
    #     # seq_len = x.size(1)
    #     # hidden_state = torch.zeros(
    #     #     (batch_size, self.rnn_cell.state_size), device=device
    #     # )
    #     outputs = []
    #     for t in range(seq_len):
    #         inputs = x[:, t]
    #         new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
    #         outputs.append(new_output)
    #     outputs = torch.stack(outputs, dim=1)  # return entire sequence