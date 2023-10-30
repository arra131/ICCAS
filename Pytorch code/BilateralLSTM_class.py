import torch
import torch.nn as nn

class BilateralLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, scope_name):  # dimension for input state, hidden state and string identifier for the cell
        super(BilateralLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scope_name = scope_name

        # define weight matrices and bias vectors for input gate, forget gate, output gate and the cell state 

        #input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        #forget gate
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

        #output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        #cell gate
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_c = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, hidden_memory_tm1, hidden_memory_tm2):   # to compute LSTM cell's output; input, tuple with previous hidden and memory state, two time steps ago

        # Unpack the hidden states and memory from the previous time steps
        previous_hidden_state, c_prev = hidden_memory_tm1
        previous_hidden_state_, _ = hidden_memory_tm2

        #calculate input gate activation 
        i = torch.sigmoid(
            self.W_i(x) +
            self.U_i(previous_hidden_state) +
            self.V_i(previous_hidden_state_)
        )

        #calculate forget gate activation 
        f = torch.sigmoid(
            self.W_f(x) +
            self.U_f(previous_hidden_state) +
            self.V_f(previous_hidden_state_)
        )

        #calculate output gate activation 
        o = torch.sigmoid(
            self.W_o(x) +
            self.U_o(previous_hidden_state) +
            self.V_o(previous_hidden_state_)
        )

        #calculate new cell state (c_)
        c_ = torch.tanh(
            self.W_c(x) +
            self.U_c(previous_hidden_state) +
            self.V_c(previous_hidden_state_)
        )

        #update the cell state (c)
        c = f * c_prev + i * c_

        #calculate hidden state 
        h_t = o * torch.tanh(c)

        #return hidden state and updated memory
        return h_t, (h_t, c)

class MultilayerCells(nn.Module):

    def __init__(self, cells):
        super(MultilayerCells, self).__init__()
        self.cells = cells # individual LSTM cells making up the multilayer cell 

    def forward(self, input, state, state_): #input, list of the previous hidden states+memory states for each individual cell, same for 2 time steps ago 
        cur_inp = input  #current input
        new_states = []   #emplty list for new states 

        for i, cell in enumerate(self.cells):
            with torch.no_grad():  # Disabling gradient computation for efficiency
                cur_inp, new_state = cell(x=cur_inp, hidden_memory_tm1=state[i], hidden_memory_tm2=state_[i])
            new_states.append(new_state)

        return cur_inp, new_states