import tensorflow as tf
import numpy as np
import os
from Contrastivelosslayer import nt_xent_loss
from utils import ones_target, zeros_target, np_sigmoid, np_rounding
from networks import C_VAE_NET
import pickle

'''my_tensor = tf.constant([[1.0,2.0,3.0,4.0], [3.0,4.0,5.0,6.0], [4.0,7.0,8.0,9.0], [8.0,9.0,0.0,1.0]])
#print(my_tensor.shape)'''

DATA_PATH = "D:\Sem\SEMESTER VII\Code"
CODE_PATH = "D:\Sem\SEMESTER VII\Code\jupyter-notebook" # come back to this later

# prepare data for training GAN
with open(os.path.join(DATA_PATH, "continuous_data.pkl"), 'rb') as f:
    data_cont = pickle.load(f)

continuous_x = data_cont

time_steps = int(continuous_x.shape[0])
#print(time_steps)
c_dim = int(continuous_x.shape[1])
#print(c_dim)
d_dim = int(continuous_x.shape[1])
#print(d_dim)

# hyper params for training

batch_size = 256
num_pre_epochs = 300
num_epochs = 500
epoch_ckpt_freq = 100
epoch_loss_freq = 10

# network size

shared_latent_dim = 15
c_z_size = shared_latent_dim
c_noise_dim = int(c_dim/2)
d_z_size = shared_latent_dim
d_noise_dim = int(d_dim/2)

# hyper-params in the networks

d_rounds=1
g_rounds=3
v_rounds=1
v_lr_pre=0.0005
v_lr=0.0001  
g_lr=0.0001
d_lr=0.0001

alpha_re = 1
alpha_kl = 0.5
alpha_mt = 0.05
alpha_ct = 0.05
alpha_sm = 1
c_beta_adv, c_beta_fm = 1, 20
d_beta_adv, d_beta_fm = 1, 10

enc_size=128
dec_size=128
enc_layers=3
dec_layers=3
keep_prob=0.9
l2scale=0.001

gen_num_units=128
gen_num_layers=3
dis_num_units=128
dis_num_layers=1
keep_prob=0.9
l2_scale=0.001
z_dim = 15

c_vae = C_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=c_dim, z_dim=c_z_size,
                  enc_size=enc_size, dec_size=dec_size, 
                  enc_layers=enc_layers, dec_layers=dec_layers, 
                  keep_prob=keep_prob, l2scale=l2scale)

c_rnn_vae_net = c_vae.build_vae(continuous_x)
print('end')