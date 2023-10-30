import torch
import torch.nn as nn
from torch.nn import init #for weight initialisation 

"""def rnn_init_state(init_, batch_size, num_layers, num_units, rnn_network=None, initial_stddev=0.02):

    if init_ == "zero":
        initial_state = (
            torch.zeros(num_layers, batch_size, num_units),
            torch.zeros(num_layers, batch_size, num_units)
        )

    elif init_ == "random":
        initial_state = (
            torch.randn(num_layers, batch_size, num_units),
            torch.randn(num_layers, batch_size, num_units)
        )

    elif init_ == "variable":
        initial_state = []
        for i in range(num_layers):
            sub_initial_state1 = nn.Parameter(torch.randn(1, batch_size, num_units) * initial_stddev)
            sub_initial_state2 = nn.Parameter(torch.randn(1, batch_size, num_units) * initial_stddev)
            initial_state.append((sub_initial_state1, sub_initial_state2))

    return initial_state
"""


def rnn_init_state(init_, batch_size, num_layers, num_units, rnn_network=None, initial_stddev=0.02):
    if init_ == "zero":
        # Initialize with zeros
        initial_state = torch.zeros(num_layers, batch_size, num_units, dtype=torch.float32)

    elif init_ == "random":
        # Initialize with random values
        initial_state = torch.normal(0.0, 1.0, size=(num_layers, 2, batch_size, num_units))
        initial_state = initial_state.unbind(0)
        initial_state = tuple([nn.LSTMStateTuple(initial_state[idx][0], initial_state[idx][1]) \
                               for idx in range(num_layers)])

    elif init_ == "variable":
        initial_state = []
        for i in range(num_layers):
            sub_initial_state1 = nn.Parameter(torch.randn(1, batch_size, num_units) * initial_stddev)
            sub_initial_state2 = nn.Parameter(torch.randn(1, batch_size, num_units) * initial_stddev)
            sub_initial_state = nn.LSTMStateTuple(sub_initial_state1, sub_initial_state2)
            initial_state.append(sub_initial_state)
        initial_state = tuple(initial_state)

    return initial_state