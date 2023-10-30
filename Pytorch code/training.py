import numpy as np
import pickle
import os
from contrastivelosslayer import nt_xent_loss
from networks import C_VAE_NET, D_VAE_NET, C_GAN_NET, D_GAN_NET
from m3gan import m3gan
from utils import renormalizer

DATA_PATH = "D:\Sem\SEMESTER VII\Code"
CODE_PATH = "D:\Sem\SEMESTER VII\Code\jupyter-notebook"

with open(os.path.join(DATA_PATH, "continuous_data.pkl"), 'rb') as f:
    data_cont = pickle.load(f)
    
with open(os.path.join(DATA_PATH, "discrete_data.pkl"), 'rb') as f:
    data_disc = pickle.load(f)

continuous_x = data_cont
discrete_x = data_disc

time_steps = continuous_x.shape[0]
c_dim = continuous_x.shape[1]
d_dim = discrete_x.shape[1]

# hyper-params for training
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

# networks for continuousGAN
c_vae = C_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=c_dim, z_dim=c_z_size,
                  enc_size=enc_size, dec_size=dec_size, 
                  enc_layers=enc_layers, dec_layers=dec_layers, 
                  keep_prob=keep_prob, l2scale=l2scale)

c_gan = C_GAN_NET(batch_size=batch_size, noise_dim=c_noise_dim, dim=c_dim,
                  gen_num_units=gen_num_units, gen_num_layers=gen_num_layers,
                  dis_num_units=dis_num_units, dis_num_layers=dis_num_layers,
                  keep_prob=keep_prob, l2_scale=l2_scale,
                  gen_dim=c_z_size, time_steps=time_steps)

# networks for discreteGAN
d_vae = D_VAE_NET(batch_size=batch_size, time_steps=time_steps, dim=d_dim, z_dim=d_z_size,
                  enc_size=enc_size, dec_size=dec_size, 
                  enc_layers=enc_layers, dec_layers=dec_layers, 
                  keep_prob=keep_prob, l2scale=l2scale)

d_gan = D_GAN_NET(batch_size=batch_size, noise_dim=d_noise_dim, dim=d_dim,
                  gen_num_units=gen_num_units, gen_num_layers=gen_num_layers,
                  dis_num_units=dis_num_units, dis_num_layers=dis_num_layers,
                  keep_prob=keep_prob, l2_scale=l2_scale,
                  gen_dim=d_z_size, time_steps=time_steps)

# create data directory for saving
checkpoint_dir = os.path.join(CODE_PATH, "data/checkpoint/")  
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

model = m3gan(batch_size=batch_size,
                  time_steps=time_steps,
                  num_pre_epochs=num_pre_epochs,
                  num_epochs=num_epochs,
                  checkpoint_dir=checkpoint_dir,
                  epoch_ckpt_freq=epoch_ckpt_freq,
                  epoch_loss_freq=epoch_loss_freq,
                  # params for c
                  c_dim=c_dim, c_noise_dim=c_noise_dim,
                  c_z_size=c_z_size, c_data_sample=continuous_x,
                  c_vae=c_vae, c_gan=c_gan,
                  # params for d
                  d_dim=d_dim, d_noise_dim=d_noise_dim,
                  d_z_size=d_z_size, d_data_sample=discrete_x,
                  d_vae=d_vae, d_gan=d_gan,
                  # params for training
                  d_rounds=d_rounds, g_rounds=g_rounds, v_rounds=v_rounds,
                  v_lr_pre=v_lr_pre, v_lr=v_lr, g_lr=g_lr, d_lr=d_lr,
                  alpha_re=alpha_re, alpha_kl=alpha_kl, alpha_mt=alpha_mt, 
                  alpha_ct=alpha_ct, alpha_sm=alpha_sm,
                  c_beta_adv=c_beta_adv, c_beta_fm=c_beta_fm, 
                  d_beta_adv=d_beta_adv, d_beta_fm=d_beta_fm)
model.build()
model.train()


