import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from init_state import rnn_init_state
from BilateralLSTM_class import BilateralLSTMCell, MultilayerCells
# import L2 regularization for pytorch

# VAE for continuous data 
class C_VAE_NET(nn.Module):
    
    def __init__(self,
                 batch_size, time_steps, dim, z_dim,
                 enc_size, dec_size, enc_layers, dec_layers,
                 keep_prob, l2scale, 
                 conditional=False, num_labels=0):
        
        super(C_VAE_NET, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim   #dimension of latent space
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.keep_prob = keep_prob   #dropout-keep probability
        self.l2scale = l2scale   #L2 regularization scale.
        self.conditional = conditional
        self.num_labels = num_labels
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_state = None

    '''def build_vae(self, input_data, conditions=None):

        if self.conditional:
            assert not self.num_labels == 0
            repeated_encoding = conditions.repeat(1, self.time_steps, 1)
            input_data_cond = torch.cat([input_data, repeated_encoding], dim=-1)
            input_enc = torch.unbind(input_data_cond, dim=1)
        else:
            input_enc = torch.unbind(input_data, dim=1)

        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = torch.zeros(self.batch_size, self.enc_size, dtype=torch.float32)
        dec_state = torch.zeros(self.batch_size, self.dec_size, dtype=torch.float32)

        # Initialize tensors
        e = torch.randn((self.batch_size, self.z_dim), dtype=torch.float32).to(input_data.device)
        c, mu, logsigma, sigma, z = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                     [0] * self.time_steps, [0] * self.time_steps

        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = torch.zeros((self.batch_size, self.dim), dtype=torch.float32).to(input_data.device)
            else:
                c_prev = c[t - 1]

            c_sigmoid = torch.sigmoid(c_prev)

            if self.conditional:
                x_hat = input_data.unbind(dim=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            # Pass through the encoder RNN
            input_rnn = torch.cat((input_enc[t], x_hat), dim=1)
            # h_enc, enc_state = self.encoder_rnn(input_rnn, enc_state)
            with torch.no_grad():
                combined_input = torch.cat([input_data[:, t, :], x_hat], dim=1)
                
                h_enc, enc_state = self.cell_enc(combined_input, enc_state)
                
                h_enc, enc_state = self.cell_enc(torch.cat([input_enc[t], x_hat], 1), enc_state)
                
            c_prev = h_enc 

            # Sampling layer
            mu[t] = torch.matmul(h_enc, w_mu) + b_mu
            logsigma[t] = torch.matmul(h_enc, w_sigma) + b_sigma
            sigma[t] = torch.exp(logsigma[t])

            # cVAE
            if self.conditional:
                z[t] = mu[t] + sigma[t] * e
                # Conditional information
                z[t] = torch.cat((z[t], conditions), dim=-1)
            else:
                z[t] = mu[t] + sigma[t] * e

            # Pass through the decoder RNN
            h_dec, dec_state = self.decoder_rnn(z[t].unsqueeze(1), dec_state)

            # Reconstruct c[t]
            self.c[t] = torch.sigmoid(torch.matmul(h_dec, self.w_h_dec) + self.b_h_dec)

        self.decoded = torch.stack(self.c, dim=1)

        return self.decoded, sigma, mu, logsigma, z'''
    
    def build_vae(self, input_data, conditions=None):
        if self.conditional:
            # cVAE
            assert not self.num_labels == 0
            repeated_encoding = torch.stack([conditions]*self.time_steps, dim=1)
            input_data_cond = torch.cat([input_data, repeated_encoding], dim=-1)
            input_enc = torch.unbind(input_data_cond, dim=1)
        else:
            input_enc = torch.unbind(input_data, dim=1)
        
        # multicell RNN -----------------------------------------------------
        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = torch.zeros(self.batch_size, dtype=torch.float32)
        dec_state = torch.zeros(self.batch_size, dtype=torch.float32)

        self.e = torch.randn((self.batch_size, self.z_dim))

        self.c, mu, logsigma, sigma, z = [0] * self.time_steps, [0] * self.time_steps, [0] * self.time_steps, \
                                        [0] * self.time_steps, [0] * self.time_steps  
        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t == 0:
                c_prev = torch.zeros((self.batch_size, self.dim))
            else:
                c_prev = self.c[t - 1]

            c_sigmoid = torch.sigmoid(c_prev)

            if self.conditional:
                x_hat = torch.unbind(input_data, dim=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid


            h_enc, enc_state = self.cell_enc(torch.cat([input_enc[t], x_hat], dim=1), enc_state )

            mu[t] = torch.matmul(h_enc, w_mu) + b_mu
            logsigma[t] = torch.matmul(h_enc, w_sigma) + b_sigma
            sigma[t] = torch.exp(logsigma[t])
            z[t] = mu[t] + sigma[t] * self.e

            h_dec, dec_state = self.cell_dec(z[t], dec_state)

            x_hat_dec = torch.matmul(h_dec, self.w_h_dec) + self.b_h_dec

            if self.conditional:
                x_hat_dec = x_hat_dec + c_sigmoid
            else:
                x_hat_dec = x_hat_dec + input_enc[t]

            self.c[t] = c_sigmoid

        return x_hat_dec, mu, logsigma, sigma, z
        
    
    def reconstruct_decoder(self, dec_input, conditions=None):
        rec_decoded = [0] * self.time_steps   #list of reconstructed decoder outputs for each time step
        rec_dec_state = self.decoder_rnn.init_hidden()   #state of decoder RNN

        for t in range(self.time_steps):
            if self.conditional:
                if conditions is not None:   #if conditions are provided
                    dec_input_with_c = torch.cat((dec_input.unbind(dim = 1)[t], conditions), dim=-1)   #concatenates with conditions tensor
                else:
                    dec_input_with_c = dec_input.unbind(dim=1)[t]   #uses current dec_input as it is 
                
                rec_h_dec, rec_dec_state = self.decoder_rnn(dec_input_with_c.unsqueeze(1), rec_dec_state)   #pass dec_input_with_c through decoder RNN

            else:
                rec_h_dec, rec_dec_state = self.decoder_rnn(dec_input.unbind(dim=1)[t].unsqueeze(1), rec_dec_state)
                
            rec_decoded[t] = torch.sigmoid(torch.matmul(rec_h_dec, self.w_h_dec)+self.b_h_dec)

        rec_decoded = torch.stack(rec_decoded, dim=1)

        return rec_decoded
    
    def buildEncoder(self):
        cell_units = []   #to store LSTM cells and dropout layers

        for num_units in range(self.enc_layers - 1):   #except the last layer 
            cell = nn.LSTMCell(self.enc_size, self.enc_size)   #input & hidden state = self.enc-size
            cell = nn.Dropout(p=1 - self.keep_prob)   #dropout layer  
            cell_units.append(cell)

        # Weight sharing in the last encoder layer
        cell = nn.LSTMCell(self.enc_size, self.enc_size)
        cell = nn.Dropout(p=1 - self.keep_prob)
        cell_units.append(cell)

        cell_enc = torch.nn.RNN(self.enc_size, self.enc_layers)
        ###nn.ModuleList(cell_units)

        return cell_enc   #returns ModuleList
    
    def buildDecoder(self):
        cell_units = []

        # Weight sharing in the first layer of the decoder
        cell = nn.LSTMCell(self.dec_size, self.dec_size)
        cell = nn.Dropout(p=1-self.keep_prob)
        cell_units.append(cell)

        for num_units in range(self.dec_layers - 1):
            cell = nn.LSTMCell(self.dec_size, self.dec_size)
            cell = nn.Dropout(p=1 - self.keep_prob)
            cell_units.append(cell)

        cell_dec = nn.ModuleList(cell_units)

        return cell_dec
    
    def weight_variable(self, shape, scope_name, name):
        with torch.no_grad():
            '''wv = nn.parameter(torch.empty(shape))
            nn.init.xavier_uniform_(wv)
            self.register_parameter(name, wv)''' # ran into some error
            wv = nn.Parameter(nn.init.xavier_normal_(torch.empty(shape), gain=1.0))

        return wv
    
    def bias_variable(self, shape, scope_name, name):
        with torch.no_grad():
            '''bv = nn.parameter(torch.empty(shape))
            nn.init.xavier_uniform_(bv)
            self.register_parameter(name, bv)'''
            bv = nn.Parameter(torch.nn.init.zeros_(torch.empty(shape)))
        return bv
    
    def buildSampling(self):
        w_mu = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_mu')
        b_mu = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
        w_sigma = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
        b_sigma = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')
        w_h_dec = self.weight_variable([self.dec_size, self.dim], scope_name='Decoder/Linear/Continuous_VAE', name='w_h_dec')
        b_h_dec = self.bias_variable([self.dim], scope_name='Decoder/Linear/Continuous_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec
    
# VAE for discrete data
class D_VAE_NET(nn.Module):
    def __init__(self,
                 batch_size, time_steps,
                 dim, z_dim,
                 enc_size, dec_size,
                 enc_layers, dec_layers,
                 keep_prob, l2scale,
                 conditional=False, num_labels=0):
        super(D_VAE_NET, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_vae(self, input_data, conditions=None):
        if self.conditional:
            #cVAE
            assert not self.num_labels == 0
            repeated_encoding = conditions.unsqueeze(1).repeat(1, self.time_steps, 1)
            input_data_cond = torch.cat([input_data, repeated_encoding], dim=-1)
            input_enc = torch.unbind(input_data_cond, dim = 1)
        else:
            input_enc = torch.unbind(input_data, dim=1)

        # multicell RNN
        self.cell_enc = self.buildEncoder()
        self.cell_dec = self.buildDecoder()
        enc_state = self.cell_enc.zero_state(self.batch_size, torch.float32)
        dec_state = self.cell_dec.zero_state(self.batch_size, torch.float32)

        self.e = torch.randn((self.batch_size, self.z_dim), dtype=torch.float32)
        mu, logsigma, sigma, z = [None]*self.time_steps, [None]*self.time_steps, \
                                 [None]*self.time_steps, [None]*self.time_steps
        
        w_mu, b_mu, w_sigma, b_sigma, self.w_h_dec, self.b_h_dec = self.buildSampling()

        for t in range(self.time_steps):
            if t==0:
                c_prev = torch.zeros((self.batch_size, self.dim), dtype=torch.float32)
            else:
                c_prev = self.c[t-1]

            c_sigmoid = torch.sigmoid(c_prev)

            if self.conditional:
                x_hat = input_data.unbind(dim=1)[t] - c_sigmoid
            else:
                x_hat = input_enc[t] - c_sigmoid

            # Encoder
            with torch.no_grad():
                h_enc, enc_state = self.cell_enc(torch.cat([input_enc[t], x_hat], dim=1), enc_state)

            # Sampling layer
            mu[t] = torch.matmul(h_enc, w_mu) + b_mu   #[z_size]
            logsigma[t] = torch.matmul(h_enc, w_sigma) + b_sigma   #[z_size]
            sigma[t] = torch.exp(logsigma[t])

            #cVAE
            if self.conditional:
                z[t] = mu[t] + sigma[t] * self.e
                #conditional information
                z[t] = torch.cat([z[t], conditions], dim=-1)
            else:
                z[t] = mu[t] + sigma[t] * self.e

            # Decoder
            with torch.no_grad():
                h_dec, dec_state = self.cell_dec(z[t], dec_state)

            self.c[t] = torch.sigmoid(torch.matmul(h_dec, self.w_h_dec) + self.b_h_dec)

        self.decoded = torch.stack(self.c, dim=1)

        return self.decoded, sigma, mu, logsigma,z
    
    def reconstruct_decoder(self, dec_input, conditions = None):
        rec_decoded = [0] * self.time_steps
        rec_dec_state = self.cell_dec.zero_state(self.batch_size, dtype = torch.float32)

        for t in range(self.time_steps):
            if self.conditional:
                if conditions is not None:
                    dec_input_with_c = torch.cat([dec_input[t], conditions], dim=-1)
                else:
                    dec_input_with_c = dec_input[t]
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input_with_c, rec_dec_state)
            else:
                rec_h_dec, rec_dec_state = self.cell_dec(dec_input[t], rec_dec_state)
            
            rec_decoded[t] = torch.sigmoid(torch.matmul(rec_h_dec, self.w_h_dec) + self.b_h_dec)

        return torch.stack(rec_decoded, dim=1)

    def buildEncoder(self):
        cell_units = []

        for num_units in range(self.enc_layers - 1):
            cell = nn.LSTMCell(self.enc_size, self.enc_size)
            cell = nn.Dropout(p=1 - self.keep_prob)
            cell_units.append(cell)

        # Weight-sharing in the last layer of encoder
        cell = nn.LSTMCell(self.enc_size, self.enc_size)
        cell = nn.Dropout(p=1 - self.keep_prob)
        cell_units.append(cell)

        cell_enc = nn.ModuleList(cell_units)

        return cell_enc
    
    def buildDecoder(self):
        cell_units = []

        # Weight-sharing in the first layer of decoder
        cell = nn.LSTMCell(self.dec_size, self.dec_size)
        cell = nn.Dropout(p=1 - self.keep_prob)
        cell_units.append(cell)

        for num_units in range(self.dec_layers - 1):
            cell = nn.LSTMCell(self.dec_size, self.dec_size)
            cell = nn.Dropout(p=1 - self.keep_prob)
            cell_units.append(cell)

        cell_dec = nn.ModuleList(cell_units)

        return cell_dec
    
    def weight_variable(self, shape, scope_name, name):
        with torch.no_grad():  
            wv = nn.parameter(torch.empty(shape))    #check alternate way torch.Tensor(*shape)
            nn.init.xavier_uniform_(wv)
            self.register_parameter(name, wv)
        return wv
    
    def bias_variable(self, shape, scope_name, name):
        with torch.no_grad():
            bv = nn.parameter(torch.empty(shape))
            nn.init.xavier_uniform_(bv)
            self.register_parameter(name, bv)
        return bv
    
    def buildSampling(self):
        w_mu = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_mu')
        b_mu = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
        w_sigma = self.weight_variable([self.enc_size, self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
        b_sigma = self.bias_variable([self.z_dim], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')
        w_h_dec = self.weight_variable([self.dec_size, self.dim], scope_name='Decoder/Linear/Continuous_VAE', name='w_h_dec')
        b_h_dec = self.bias_variable([self.dim], scope_name='Decoder/Linear/Continuous_VAE', name='b_h_dec')

        return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec
    
class C_GAN_NET(nn.Module):
    def __init__(self, batch_size, noise_dim, dim, 
                 gen_dim, time_steps,
                 gen_num_units, gen_num_layers,
                 dis_num_units, dis_num_layers,
                 keep_prob, l2_scale,
                 conditional=False, num_labels=0):
        super(C_GAN_NET, self).__init__()

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dim = dim
        self.gen_dim = gen_dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.keep_prob = keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_GenRNN(self, input_noise, conditions = None):
        if self.conditional:
            # create input noise for the generator
            repeated_encoding = conditions.repeat(1, self.time_steps, 1)   #replicates the conditions self.time_steps times along the second dimension
            noise_with_c = torch.cat((input_noise, repeated_encoding), dim=2)
            self.g_input = torch.unbind(noise_with_c, dim=1)

            # create multi-cell LSTM layers 
            cells_list = nn.ModuleList()
            for i in range(self.gen_num_layers):
                g_input_dim = (self.noise_dim + self.num_labels) if i == 0 else self.gen_num_units
                bilstmcell = BilateralLSTMCell(input_dim=g_input_dim, hidden_dim=self.gen_num_units, scope_name="Continuous_generator/RNNCell_%d" % i)
                cells_list.append(bilstmcell)
            self.g_rnn_network = nn.ModuleList(cells_list)

        else:
            # create input noise for the generator
            self.g_input = torch.unbind(input_noise, dim=1)

            # create multi-cell LSTM layers
            for i in range(self.gen_num_layers):
                g_input_dim = self.noise_dim if i == 0 else self.gen_num_units
                bilstmcell = BilateralLSTMCell(input_dim=g_input_dim, hidden_dim=self.gen_num_units, scope_name="Continuous_generator/RNNCell_%d" % i)
                cells_list.append(bilstmcell)
            self.g_rnn_network = nn.ModuleList(cells_list)

        # create initial state for multi-cell LSTM
        initial_state = []

        for i in range(self.gen_num_layers):
            state_ = torch.stack([torch.zeros(self.batch_size, self.gen_num_units),
                                 torch.zeros(self.batch_size, self.gen_num_units)], dim=1)
            initial_state.append(state_)

        return initial_state

    def gen_Onestep(self, t, state):
        with torch.no_grad():
            cell_outputs = []   #to store outputs from each LSTM layer
            
            for i in range(self.gen_num_layers):
                # Get the current LSTM cell from ModuleList
                lstm_cell = self.g_rnn_network[i]  

                # Calculate cell output and new state for the current layer
                cell_new_output_, new_state = lstm_cell(self.g_input[t], state[i])

                # Apply a linear layer with sigmoid activation
                new_output_linear = torch.sigmoid(F.linear(cell_new_output_, weight=lstm_cell.weight_hh, bias=lstm_cell.bias_hh))

                # Store output of current layer
                cell_outputs.append(new_output_linear)

                # Update the state for the next layer
                if i < self.gen_num_layers - 1:
                    state[i + 1] = new_state

            final_output = torch.cat(cell_outputs, dim=-1)

        return final_output, state
    
    def build_Discriminator(self, input_discriminator):
        with torch.no_grad():
            # Define a list to store LSTM cell layers
            cell_units = []

            for num_units in range(self.dis_num_layers):
                # Create a LSTM cell
                lstm_cell = nn.LSTMCell(self.dis_num_units)

                # Apply dropout 
                lstm_cell = nn.Dropout(lstm_cell, p=1-self.keep_prob)

                cell_units.append(lstm_cell)

            # Create a multi layer LSTM network
            d_rnn_network = nn.ModuleList(cell_units)

            # Initialize the hidden states and cell states
            initial_state = []
            for i in range(self.dis_num_layers):
                state_ = (torch.zeros(self.batch_size, self.dis_num_units), torch.zeros(self.batch_size, self.dis_num_units))
                initial_state.append(state_)

            # Initialize a list to store outputs from each time step
            outputs = []

            for input_step in input_discriminator:
                state = initial_state   #reset at each time step

                for i in range(self.dis_num_layers):
                    # Get the current LSTM cell from the ModuleList
                    lstm_cell = d_rnn_network[i]

                    # Cell output and new state for current layer
                    cell_new_output, new_state = lstm_cell(input_step, state[i])

                    # Update the state for next layer
                    state[i] = new_state

                # Store the output of last LSTM layer for each time step
                outputs.append(cell_new_output)

            # Stack the outputs along time step dimension
            outputs = torch.stack(outputs, dim=1)

            # Flatten the output and pass through a dense layer
            flattened_output = outputs.view(-1, self.dis_num_units)
            result = nn.Linear(self.dis_num_units, 1)(flattened_output)

        return result, outputs
    
class D_GAN_NET(nn.Module):
    def __init__(self, batch_size, noise_dim, gen_dim, dim,
                 time_steps, gen_num_units, gen_num_layers,
                 dis_num_units, dis_num_layers, keep_prob,
                 l2_scale, conditional=False, num_labels=0):
        super(D_GAN_NET, self).__init__()

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.gen_dim = gen_dim
        self.dim = dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.keep_prob = keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels

    def build_GenRNN(self, input_noise, conditions=None):
        if self.conditional:
            # Create an input noise for the generator
            repeated_encoding = conditions.repeat(1, self.time_steps, 1)
            noise_with_c = torch.cat((input_noise, repeated_encoding), dim = 2)
            self.g_input = torch.unbind(noise_with_c, dim=1)

            # Create a multi-cell LSTM for the generator
            cells_lsits = []
            for i in range(self.gen_num_layers):
                g_input_dim = (self.noise_dim + self.num_labels) if i == 0 else self.gen_num_units
                lstm_cell = torch.nn.LSTMCell(input_size=g_input_dim, hidden_size=self.gen_num_units)
                cells_lsits.append(lstm_cell)
            self.g_rnn_network = torch.nn.ModuleList(cells_lsits)

        else:
            # Create an input noise for the generator
            self.g_input = torch.unbind(input_noise, dim=1)

            # Create a multi-cell LSTM for the generator
            cells_lsits = []
            for i in range(self.gen_num_layers):
                g_input_dim = self.noise_dim if i == 0 else self.gen_num_units
                lstm_cell = torch.nn.LSTMCell(input_size=g_input_dim, hidden_size=self.gen_num_units)
                cells_lsits.append(lstm_cell)
            self.g_rnn_network = torch.nn.ModuleList(cells_lsits)

        # Initial state for multi-cell LSTM
        initial_state = []
        for i in range(self.gen_num_layers):
            state_ = (torch.zeros(self.batch_size, self.gen_num_units), torch.zeros(self.batch_size, self.gen_num_units))
            initial_state.append(state_)

        return initial_state
    
    def gen_Onestep(self, t, state):
        with torch.no_grad():
            # List to store outputs from each LSTM layer
            cell_outputs = []

            for i in range(self.gen_num_layers):
                # Current LSTM cell from ModuleList
                lstm_cell = self.g_rnn_network[i]

                # Cell o/p & new state for current layer
                cell_new_output_, new_state = lstm_cell(self.g_input[t], state[i])

                # Linear layer with sigmoid activation
                new_output_linear = torch.sigmoid(F.linear(cell_new_output_, weight=lstm_cell.weight_hh, bias=lstm_cell.bias_hh))

                # Store output of current layer
                cell_outputs.append(new_output_linear)

                # Update the state for next layer
                if i < self.gen_num_layers - 1:
                    state[i + 1] = new_state

            final_output = torch.cat(cell_outputs, dim=-1)

        return final_output, state

    def build_Discriminator(self, input_discriminator):
        with torch.no_grad():
            # Define list to store LSTM cell layers
            cell_units = []

            for num_units in range(self.dis_num_layers):
                # Create a LSTM cell
                lstm_cell = nn.LSTMCell(self.dis_num_units)

                # Apply dropout
                lstm_cell = nn.Dropout(lstm_cell, p=1-self.keep_prob)

                cell_units.append(lstm_cell)

            # Create a multi layer LSTM network
            d_rnn_network = nn.ModuleList(cell_units)

            # Initialize hidden states and cell states
            initial_state = []
            for i in range(self.dis_num_layers):
                state_ = (torch.zeros(self.batch_size, self.dis_num_units),
                          torch.zeros(self.batch_size, self.dis_num_units))
                initial_state.append(state_)

            # Initialize a list to store all outputs from each time step
            outputs = []

            for input_step in input_discriminator:
                state = initial_state   #reset at each time step

                for i in range(self.dis_num_layers):
                    # Get current cell from ModuleList
                    lstm_cell = d_rnn_network[i]

                    # Cell output and new state for current layer
                    cell_new_output, new_state = lstm_cell(input_step, state[i])

                    # Update the state for next layer
                    state[i] = new_state

                outputs.append(cell_new_output)

            outputs = torch.stack(outputs, dim=1)

            # Flatten output and pass through dense layer
            flattened_output = outputs.view(-1, self.dis_num_units)
            result = nn.Linear(self.dis_num_units, 1)(flattened_output)

        return result