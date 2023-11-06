import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from Bilateral_lstm_class import Bilateral_LSTM_cell, MultilayerCells
import pickle
import os

DATA_PATH = "D:\Sem\SEMESTER VII\Code"
CODE_PATH = "D:\Sem\SEMESTER VII\Code\jupyter-notebook" # come back to this later

with open(os.path.join(DATA_PATH, "continuous_data.pkl"), 'rb') as f:
    data_cont = pickle.load(f)

continuous_x = data_cont

# timeseries params:

time_steps = continuous_x.shape[0]
c_dim = continuous_x.shape[1]
d_dim = continuous_x.shape[1]

def buildEncoder():
    cell_units = []   #to store LSTM cells and dropout layers

    for num_units in range(3-1):   #except the last layer 
        cell = nn.LSTMCell(128, 128)   #input & hidden state = self.enc-size
        cell = nn.Dropout(p=1 - 0.9)   #dropout layer  
        cell_units.append(cell)

    # Weight sharing in the last encoder layer
    cell = nn.LSTMCell(128, 128)
    cell = nn.Dropout(p=1 - 0.9)
    cell_units.append(cell)

    cell_enc = torch.nn.RNN(128, 3)
    ###nn.ModuleList(cell_units)

    return cell_enc   #returns ModuleList

def weight_variable(shape, scope_name, name):
    with torch.no_grad():
        wv = nn.Parameter(nn.init.xavier_normal_(torch.empty(shape), gain=1.0))
    return wv

def bias_variable(shape, scope_name, name):
    with torch.no_grad():
        bv = nn.Parameter(torch.nn.init.zeros_(torch.empty(shape)))
    return bv

def buildSampling():
    w_mu = weight_variable([128, 15], scope_name='Sampling_layer/Shared_VAE', name='w_mu')
    b_mu = bias_variable([15], scope_name='Sampling_layer/Shared_VAE', name='b_mu')
    w_sigma = weight_variable([128, 15], scope_name='Sampling_layer/Shared_VAE', name='w_sigma')
    b_sigma = bias_variable([15], scope_name='Sampling_layer/Shared_VAE', name='b_sigma')
    w_h_dec = weight_variable([128, c_dim], scope_name='Decoder/Linear/Continuous_VAE', name='w_h_dec')
    b_h_dec = bias_variable([c_dim], scope_name='Decoder/Linear/Continuous_VAE', name='b_h_dec')

    return w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec

def build_vae(input_data):
    input_enc = input_data
    cell_enc = buildEncoder()
    enc_state = torch.zeros(256, dtype=torch.float32)

    e = torch.randn((256, 3))
    c, mu, logsigma, sigma, z = [0] * time_steps, [0] * time_steps, [0] * time_steps, \
                                        [0] * time_steps, [0] * time_steps  
    w_mu, b_mu, w_sigma, b_sigma, w_h_dec, b_h_dec = buildSampling()

    for t in range(time_steps):
        print(t)
        if t == 0:
            c_prev = torch.zeros((256, 3))
            print('if c is', c_prev.shape)
        else:
            c_prev = c[t-1]
            print('else c is', c_prev.shape)

        c_sigmoid = torch.sigmoid(c_prev)
        print(c_sigmoid.shape)

        x_hat = input_enc[t] - c_sigmoid

        print(input_enc.shape)
        print(x_hat.shape)
        h_enc, enc_state = cell_enc(torch.cat([input_enc, x_hat], dim=0), enc_state)
    
    return x_hat

print(type(continuous_x))
ts = continuous_x.values
cont_data = torch.tensor(ts)
print(type(cont_data))
a = build_vae(cont_data)
print('end')
 