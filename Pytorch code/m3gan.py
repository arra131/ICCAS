import torch
import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import os
from contrastivelosslayer import nt_xent_loss
from utils import ones_target, zeros_target, np_rounding, np_sigmoid
from visualise import visualise_gan, visualise_vae

class m3gan(object):

    def __init__(self,
                 #--shared params:
                 batch_size, time_steps,
                 num_pre_epochs, num_epochs,
                 checkpoint_dir, epoch_ckpt_freq, epoch_loss_freq,
                 #--params for c
                 c_dim, c_noise_dim,
                 c_z_size, c_data_sample,
                 c_gan, c_vae,
                 #--params for d
                 d_dim, d_noise_dim,
                 d_z_size, d_data_sample,
                 d_gan, d_vae,
                 #--params for training
                 d_rounds, g_rounds, v_rounds,
                 v_lr_pre, v_lr, g_lr, d_lr,
                 alpha_re, alpha_kl, alpha_mt,
                 alpha_ct, alpha_sm, 
                 c_beta_adv, c_beta_fm,
                 d_beta_adv, d_beta_fm,
                 #--label information
                 conditional=False, num_labels=0, statics_label=None): 
        
        super(m3gan, self).__init__()
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_pre_epochs = num_pre_epochs
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.epoch_ckpt_freq = epoch_ckpt_freq
        self.epoch_loss_freq = epoch_loss_freq

        # params for continuous 
        self.c_dim = c_dim
        self.c_noise_dim = c_noise_dim
        self.c_z_size = c_z_size
        self.c_data_sample = c_data_sample
        self.c_rnn_vae_net = c_vae
        self.cgan = c_gan

        # params for discrete
        self.d_dim = d_dim
        self.d_noise_dim = d_noise_dim
        self.d_z_size = d_z_size
        self.d_data_sample = d_data_sample
        self.d_rnn_vae_net = d_vae
        self.dgan = d_gan

        #params for training
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.v_rounds = v_rounds

        # params for learning rate 
        self.v_lr_pre = v_lr_pre
        self.v_lr = v_lr
        self.g_lr = g_lr
        self.d_lr = d_lr

        # params for loss scalar
        self.alpha_re = alpha_re
        self.alpha_kl = alpha_kl
        self.alpha_mt = alpha_mt
        self.alpha_ct = alpha_ct
        self.alpha_sm = alpha_sm
        self.c_beta_adv = c_beta_adv
        self.c_beta_fm = c_beta_fm
        self.d_beta_adv = d_beta_adv
        self.d_beta_fm = d_beta_fm
        
        # params for label information 
        self.num_labels = num_labels
        self.conditional = conditional
        self.statics_label = statics_label

    def build(self):
        self.build_tf_graph()
        self.build_loss()
        self.build_summary()
        self.save()

    def save(self, global_id, model_name=None, checkpoint_dir=None):
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_{global_id}.pth')
        torch.save(self, checkpoint_path)

    def load(self, model_name=None, checkpoint_dir=None):
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
        model = torch.load(checkpoint_path)
        global_id = int(model_name.split("_")[-1])
        return global_id
    
    def build_tf_graph(self):
        # STEP 1 - VAE TRAINING -------------------------------------------------------
        ## pretrain vae for C

        if self.conditional:
            self.real_data_label_pl = torch.zeros((self.batch_size, self.num_labels), dtype=torch.float32, requires_grad=False, name = "continuous_real_data")

        shape1 = (self.batch_size, self.time_steps, self.c_dim)
        self.c_real_data_pl = torch.zeros(shape1, dtype=torch.float32, requires_grad=False)
        print("done")

        if self.conditional:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl, self.real_data_label_pl)     #check for relation with networks.py
        else:
            self.c_decoded_output, self.c_vae_sigma, self.c_vae_mu, self.c_vae_logsigma, self.c_enc_z = \
                self.c_rnn_vae_net.build_vae(self.c_real_data_pl)

        ## add validation set here
        self.c_vae_test_data_pl = torch.zeros((self.batch_size, self.time_steps, self.c_dim), dtype=torch.float32, requires_grad=False, name = "vae_validation_c_data")

        if self.conditional:
            self.c_vae_test_decoded, _, _, _, _ = self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.c_vae_test_decoded, _, _, _, _ = self.c_rnn_vae_net.build_vae(self.c_vae_test_data_pl)

        ## pretrain vae for d
        self.d_real_data_pl = torch.zeros((self.batch_size, self.time_steps, self.d_dim), dtype=torch.float32, requires_grad=False, name = "discrete_real_data")
        if self.conditional:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl, self.real_data_label_pl)
        else:
            self.d_decoded_output, self.d_vae_sigma, self.d_vae_mu, self.d_vae_logsigma, self.d_enc_z = \
                self.d_rnn_vae_net.build_vae(self.d_real_data_pl)
            
        ## add validation set here
        self.d_vae_test_data_pl = torch.zeros((self.batch_size, self.time_steps, self.d_dim), dtype=torch.float32, requires_grad=False, name="vae_validation_d_data")

        if self.conditional:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl, self.real_data_label_pl)
        else:
            self.d_vae_test_decoded, _, _, _, _ = \
                self.d_rnn_vae_net.build_vae(self.d_vae_test_data_pl)

        # STEP 2 - GENERATOR TRAINING--------------------------------------------------------------------------------------------

        ## cgan - initialisation
        self.c_gen_input_noise_pl = torch.zeros((None, self.time_steps, self.c_noise_dim), dtype=torch.float32, requires_grad=False, name="continuous_generator_input_noise")
        if self.conditional:
            c_initial_state = self.cgan.build_GenRNN(self.c_gen_input_noise_pl, self.real_data_label_pl)
        else:
            c_initial_state = self.cgan.build_GenRNN(self.c_gen_input_noise_pl)

        ## dgan - initialisation
        self.d_gen_input_noise_pl = torch.zeros((None, self.time_steps, self.d_noise_dim), dtype=torch.float32, requires_grad=False, name="discrete_generator_input_noise")
        if self.conditional:
            d_initial_state = self.dgan.build_GenRNN(self.d_gen_input_noise_pl, self.real_data_label_pl)
        else:
            d_initial_state = self.dgan.build_GenRNN(self.d_gen_input_noise_pl)

        ### sequentially coupled training steps
        self.c_gen_output_latent = []
        self.d_gen_output_latent = []

        d_new_state = d_initial_state
        c_new_state = c_initial_state

        for t in range(self.time_steps):
            d_state = d_new_state if t > 0 else d_initial_state
            c_state = c_new_state if t > 0 else c_initial_state


            # new_state is a tuple of (h_i, c_i)
            d_new_linear, d_new_state = self.dgan.gen_Onestep(t, [d_state, c_state])
            c_new_linear, c_new_state = self.dgan.gen_Onestep(t, [c_state, d_state])

            self.d_gen_output_latent.append(d_new_linear)
            self.c_gen_output_latent.append(c_new_linear)

        # STEP 3: DECODER ---------------------------------------------------------------------------------------

        ## dgan - decoding 
        if self.conditional:
            self.d_gen_decoded = self.d_rnn_vae_net.reconstruct_decoder(dec_input = self.d_gen_output_latent,
                                                                        conditions = self.real_data_label_pl)
        else:
            self.d_gen_decoded = self.d_rnn_vae_net.reconstruct_decoder(dec_input = self.d_gen_output_latent)
        self.d_gen_decoded = torch.unbind(self.d_gen_decoded, dim=1)

        ## cgan decoding 
        if self.conditional:
            self.c_gen_decoded = self.c_rnn_vae_net.reconstruct_decoder(dec_input = self.c_gen_output_latent,
                                                                        conditions = self.real_data_label_pl)
        else:
            self.c_gen_decoded = self.c_rnn_vae_net.reconstruct_decoder(dec_input = self.c_gen_output_latent)
        self.c_gen_decoded = torch.unbind(self.c_gen_decoded, dim=1)

        # STEP 4: DISCRIMINATOR---------------------------------------------------------------------------------------------------------

        ## cgan - discriminator
        self.c_fake, self.c_fake_fm = self.cgan.build_Discriminator(self.d_gen_decoded)
        self.d_real, self.c_real_fm = self.cgan.build_Discriminator(torch.unbind(self.c_real_data_pl, dim=1))

        ## dgan - discrimator
        self.d_fake = self.dgan.build_Discriminator(self.d_gen_decoded)
        self.d_real = self.dgan.build_Discriminator(torch.unbind(self.d_real_data_pl, dim=1))

    def build_loss(self):

        #################
        # (1) VAE LOSS #
        ################
        alpha_re = self.alpha_re
        alpha_kl = self.alpha_kl
        alpha_mt = self.alpha_mt
        alpha_ct = self.alpha_ct
        alpha_sm = self.alpha_sm

        # ** 1. VAE loss for c **
        self.c_re_loss = torch.nn.MSELoss(self.c_real_data_pl, self.c_decoded_output) #mean squared error loss

        c_kl_loss = [] * self.time_steps #KL divergence loss
        for t in range(self.time_steps):
            c_kl_loss[t] = 0.5 * (
                torch.sum(self.c_vae_sigma[t], dim=1) +
                torch.sum(self.c_vae_mu[t].pow(2), dim=1) - 
                torch.sum(torch.exp(self.c_vae_logsigma[t]) + 1, dim=1) 
            )
            self.c_kl_loss.append(c_kl_loss)

        # Mean of KL divergence
        self.c_kl_loss = torch.mean(torch.sum(self.c_kl_loss, dim=0)) 

        # ** 2. EUCLIDEAN distance between latent representation from d and c **
        x_latent_1 = torch.stack(self.c_enc_z, dim=1)
        x_latent_2 = torch.stack(self.d_enc_z, dim=1)
        self.vae_matching_loss = torch.nn.MSELoss(x_latent_1, x_latent_2)

        """ in case euclidean(L2)
        # Assuming self.c_enc_z and self.d_enc_z are lists of tensors
        x_latent_1 = torch.stack(self.c_enc_z, dim=1)
        x_latent_2 = torch.stack(self.d_enc_z, dim=1)

        # Calculate Euclidean distance (L2 norm) between the stacked tensors
        diff = x_latent_1 - x_latent_2
        euclidean_distance = torch.norm(diff, p=2, dim=-1)  # Assuming the last dimension represents the latent space dimensions

        # Calculate the mean squared error as the matching loss
        vae_matching_loss = torch.mean(euclidean_distance.pow(2))

        # You can use vae_matching_loss in your model or training loop as needed."""

        # ** 3. Contrastive loss **
        self.vae_contra_loss = nt_xent_loss(torch.reshape(x_latent_1, (x_latent_1.shape(0), -1)),
                                            torch.reshape(x_latent_2, (x_latent_2.shape(0)), self.batch_size))
        
        # ** 4. If label: conditional VAE and classification cross entropy loss
        if self.conditional:
            # exclude the label info from the latent vector
            x_latent_1 = x_latent_1[:, :, :-1]
            x_latent_2 = x_latent_2[:, :, :-1]

            # with variable scope

            # concatenate the input tensor
            concatenated_input = torch.cat([x_latent_1, x_latent_2], dim=1)
            vae_flatten_input = torch.nn.Flatten(concatenated_input, start_dim=1)

            # create a linear layer with ReLU activation
            vae_hidden = torch.nn.Linear(vae_flatten_input.size(1), 24)
            vae_hidden_layer = F.relu(vae_hidden(vae_flatten_input))

            # Create another linear layer with tanh activation
            vae_logits_layer = torch.nn.Linear(24, 4)
            vae_logits = torch.tanh(vae_logits_layer(vae_hidden_layer))

            criterion = torch.nn.CrossEntropyLoss()

            self.vae_semantics_loss = criterion(vae_logits, self.real_data_label_pl)

        if self.conditional:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
            
        else:
            self.c_vae_loss = alpha_re * self.c_re_loss + \
                              alpha_kl * self.c_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss
            
        # vae validation loss 
        self.c_vae_valid_loss = torch.nn.MSELoss(self.c_vae_test_data_pl, self.c_vae_test_decoded)

        # VAE loss for D (BCE loss)
        bce_criterion = torch.nn.BCELoss()
        self.d_re_loss = bce_criterion(self.d_decoded_output, self.d_real_data_pl)

        d_kl_loss = [0] * self.time_steps # KL divergence
        for t in range(self.time_steps):
            kl_term = 0.5 * (torch.sum(self.d_vae_sigma[t], dim = 1) +
                                  torch.sum(self.d_vae_mu[t].pow(2), dim = 1) - 
                                  torch.sum(self.d_vae_logsigma[t] + 1, dim = 1))
            d_kl_loss[t] = kl_term

        # total kl divergence 
        self.d_kl_loss = 0.1 * torch.mean(torch.stack(d_kl_loss))

        if self.conditional:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss + \
                              alpha_sm * self.vae_semantics_loss
        else:
            self.d_vae_loss = alpha_re * self.d_re_loss + \
                              alpha_kl * self.d_kl_loss + \
                              alpha_mt * self.vae_matching_loss + \
                              alpha_ct * self.vae_contra_loss
            
        # vae validation loss
        self.d_vae_valid_loss = torch.nn.MSELoss(self.d_vae_test_data_pl, self.d_vae_test_decoded)

        ##########################
        # (2) DISCRIMINATOR LOSS #
        ##########################

        # CGAN - discriminator loss (no activation function for discriminator therefore from logits)
        def ones_target(batch_size, min_val, max_val):
            return torch.rand(batch_size) * (max_val - min_val) + min_val
        
        def zeros_target(batch_size, min_val, max_val):
            return torch.rand(batch_size) * (max_val - min_val) + min_val
        
        real_target_c = ones_target(self.batch_size, 0.7, 1.2)
        fake_target_c = zeros_target(self.batch_size, 0.1, 0.3)

        bce_logit_criterion = torch.nn.BCEWithLogitsLoss()

           # Losses for real and fake samples 
        self.continuous_d_loss_real = torch.mean(F.binary_cross_entropy_with_logits(self.c_real, real_target_c))
        self.continuous_d_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(self.c_fake, fake_target_c))
        self.continuous_d_loss = self.continuous_d_loss_real + self.continuous_d_loss_fake

        # DGAN - discriminator loss
        real_target_d = ones_target(self.batch_size, 0.8, 0.9)
        fake_target_d = zeros_target(self.batch_size, 0.1, 0.1)

        self.discrete_d_loss_real = torch.mean(F.binary_cross_entropy_with_logits(self.d_real, real_target_d))
        self.discrete_d_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(self.d_real, fake_target_d))
        self.dicrete_d_loss  = self.discrete_d_loss_real + self.discrete_d_loss_fake

        ######################
        # (2) GENERATOR LOSS #
        ######################

        # CGAN - adversarial loss
        self.c_gen_loss_adv = torch.mean(
            F.binary_cross_entropy_with_logits(self.c_fake, torch.ones_like(self.c_fake)))
        
        # cgan feature matching 
        c_fake_fm_mean, c_fake_fm_var = torch.mean(self.c_fake_fm, dim=0)
        c_real_fm_mean, c_real_fm_var = torch.mean(self.c_real_fm, dim=0)

        self.c_g_loss_v1 = torch.mean(torch.abs(torch.sqrt(c_fake_fm_var + 1e-6) - torch.sqrt(c_real_fm_var + 1e-6)))
        self.c_g_loss_v2 = torch.mean(torch.abs(torch.sqrt(torch.abs(c_fake_fm_mean)) - torch.sqrt(torch.abs(c_real_fm_mean))))

        self.c_gen_loss_fm = self.c_g_loss_v1 + self.c_g_loss_v2

        # cgan - add two losses for generator
        c_beta_adv = self.c_beta_adv
        c_beta_fm = self.c_beta_fm
        self.c_gen_loss = c_beta_adv * self.c_gen_loss_adv + c_beta_fm * self.c_gen_loss_fm

        # DGAN - adversarial loss
        self.d_gen_loss_adv = torch.mean(
            F.binary_cross_entropy_with_logits(self.d_fake, torch.ones_like(self.d_fake)))
        
        # dgan - feature matching (statistical comparison between generated and real data)
        d_gen_decoded_stack = torch.stack(self.d_gen_decoded, dim=1)
        d_gen_decoded_mean, d_gen_decoded_var = torch.mean(d_gen_decoded_stack, dim=0)
        d_real_data_pl_mean, d_real_data_pl_var = torch.mean(self.d_real_data_pl, dim=0)

        self.d_g_loss_v1 = torch.mean(torch.abs(torch.sqrt(d_gen_decoded_var + 1e-6) - torch.sqrt(d_real_data_pl_var + 1e-6)))
        self.d_g_loss_v2 = torch.mean(torch.abs(d_gen_decoded_mean - d_real_data_pl_mean))
        self.d_gen_loss_fm = self.d_g_loss_v1 + self.d_g_loss_v2

        # dgan - add two losses for generator
        d_beta_adv = self.d_beta_adv
        d_beta_fm = self.d_beta_fm
        self.d_gen_loss = d_beta_adv * self.d_gen_loss_adv + d_beta_fm * self.d_gen_loss_fm

        ##################
        # (3) OPTIMIZER #
        ##################

        t_vars = torch.nn.parameter()
        c_vae_vars = [var for var in t_vars if 'Continuous_VAE' in var.name]
        d_vae_vars = [var for var in t_vars if 'Discrete_VAE' in var.name]
        s_vae_vars = [var for var in t_vars if 'Shared_VAE' in var.name]
        c_g_vars = [var for var in t_vars if 'Continuous_generator' in var.name]
        c_d_vars = [var for var in t_vars if 'Continuous_discriminator' in var.name]
        d_g_vars = [var for var in t_vars if 'Discrete_generator' in var.name]
        d_d_vars = [var for var in t_vars if 'Discrete_discriminator' in var.name]

        # Optimizer for c of vae
        self.c_v_op_pre = optim.Adam([{'params': c_vae_vars + s_vae_vars, 'lr': self.v_lr_pre}])
        self.c_v_op_pre.zero_grad()
        self.c_vae_loss.backward()
        self.c_v_op_pre.step()

        # Optimizer for d of vae
        self.d_v_op_pre = optim.Adam([{'params': d_vae_vars + s_vae_vars, 'lr': self.v_lr_pre}])
        self.d_v_op_pre.zero_grad()
        self.d_vae_loss.backward()
        self.d_v_op_pre.step()

        # Optimizer for c of vae
        self.c_v_op = optim.Adam([{'params': c_vae_vars + s_vae_vars, 'lr': self.v_lr}])
        self.c_v_op.zero_grad()
        self.c_vae_loss.backward()
        self.c_v_op.step()

        # Optimizer for d of vae
        self.d_v_op = optim.Adam([{'params': d_vae_vars + s_vae_vars, 'lr': self.v_lr}])
        self.d_v_op.zero_grad()
        self.d_vae_loss.backward()
        self.d_v_op.step()

        # Optimizer for c of generator
        self.c_g_op = optim.Adam([{'params': c_g_vars , 'lr': self.g_lr}])
        self.c_g_op.zero_grad()
        self.c_gen_loss.backward()
        self.c_g_op.step()

        # Optimizer for d of generator
        self.d_g_op = optim.Adam([{'params': d_g_vars , 'lr': self.g_lr}])
        self.d_g_op.zero_grad()
        self.d_gen_loss.backward()
        self.d_g_op.step()
        
        # Optimizer for c of discriminator
        self.c_d_op = optim.Adam([{'params': c_d_vars , 'lr': self.d_lr}])
        self.c_d_op.zero_grad()
        self.continuous_d_loss.backward()
        self.c_d_op.step()

        # Optimizer for d of discriminator
        self.d_d_op = optim.Adam([{'params': d_d_vars , 'lr': self.d_lr}])
        self.d_d_op.zero_grad()
        self.dicrete_d_loss.backward()
        self.d_d_op.step()

    """def build_summary(self):

        # create a log for Summary writing for c_vae
        self.c_vae_summary  = SummaryWriter()

        # loss summary of variational autoencoder for c
        self.c_vae_summary.add_scalar("C_VAE_loss/reconstruction_loss", self.c_re_loss, global_step=0)
        self.c_vae_summary.add_scalar("C_VAE_loss/kl_divergence_loss", self.c_kl_loss, global_step=0)
        self.c_vae_summary.add_scalar("C_VAE_loss/matching_loss", self.vae_matching_loss, global_step=0)
        self.c_vae_summary.add_scalar("C_VAE_loss/contrastive_loss", self.vae_contra_loss, global_step=0)
        if self.conditional:
            self.c_vae_summary.add_scalar("C_VAE_loss/semantic_loss", self.vae_semantics_loss, global_step=0)
        self.c_vae_summary.add_scalar("C_VAE_loss/vae_loss", self.c_vae_loss, global_step=0)
        self.c_vae_summary.add_scalar("C_VAE_loss/validation_loss", self.c_vae_valid_loss, global_step=0)

        # Close the SummaryWriter
        self.c_vae_summary.close()

        # create a log for Summary writing for d_vae
        self.d_vae_summary = SummaryWriter()

        # loss summary of variational autoencoder for d
        self.d_vae_summary.add_scalar("D_VAE_loss/reconstruction_loss", self.d_re_loss, global_step=0)
        self.d_vae_summary.add_scalar("D_VAE_loss/kl_divergence_loss", self.d_kl_loss, global_step=0)
        self.d_vae_summary.add_scalar("D_VAE_loss/matching_loss", self.vae_matching_loss, global_step=0)
        self.d_vae_summary.add_scalar("D_VAE_loss/contrastive_loss", self.vae_contra_loss, global_step=0)
        if self.conditional:
            self.d_vae_summary.add_scalar("D_VAE_loss/semantic_loss", self.vae_semantics_loss, global_step=0)
        self.d_vae_summary.add_scalar("D_VAE_loss/vae_loss", self.d_vae_loss, global_step=0)
        self.d_vae_summary.add_scalar("D_VAE_loss/validation_loss", self.d_vae_valid_loss, global_step=0)

        # Close the SummaryWriter
        self.d_vae_summary.close()

        # create a log for Summary writing for c discriminator
        self.c_discriminator_summary = SummaryWriter()

        # loss summary of discriminator for c
        self.c_discriminator_summary.add_scalar("c_discriminator_loss/d_real", self.continuous_d_loss_real, global_step=0)
        self.c_discriminator_summary.add_scalar("c_discriminator_loss/d_fake", self.continuous_d_loss_fake, global_step=0)
        self.c_discriminator_summary.add_scalar("c_discriminator_loss/discriminator_loss", self.continuous_d_loss, global_step=0)

        # Close the SummaryWriter
        self.c_discriminator_summary.close()
        
        #create a log for Summary writing for c generator 
        self.c_generator_summary = SummaryWriter()

        # loss summary of generator for c
        self.c_generator_summary.add_scalar("c_generator_loss/adversarial_loss", self.c_gen_loss_adv, global_step=0)
        self.c_generator_summary.add_scalar("c_generator_loss/feature_matching_loss_v1", self.c_g_loss_v1, global_step=0)
        self.c_generator_summary.add_scalar("c_generator_loss/feature_matching_loss_v2", self.c_g_loss_v2, global_step=0)
        self.c_generator_summary.add_scalar("c_generator_loss/feature_matching_loss", self.c_gen_loss_fm, global_step=0)
        self.c_generator_summary.add_scalar("c_generator_loss/generator_loss", self.c_gen_loss, global_step=0)
        self.c_generator_summary.close()

        # create a log for Summary writing for d discriminator
        self.d_discriminator_summary = SummaryWriter()

        # loss summary of discriminator for d
        self.d_discriminator_summary.add_scalar("d_discriminator_loss/dicrete_d_loss_real", self.dicrete_d_loss_real)
        self.d_discriminator_summary.add_scalar("d_discriminator_loss/dicrete_d_loss_fake", self.dicrete_d_loss_fake)
        self.d_discriminator_summary.add_scalar("d_discriminator_loss/d_discriminator", self.dicrete_d_loss)
        self.d_discriminator_summary.close()

        # create a log for Summary writing for d generator
        self.d_generator_summary = SummaryWriter()

        # loss summary of generator for d
        self.d_generator_summary.add_scalar("d_generator_loss/g_loss_v1", self.d_g_loss_v1, global_step=0)
        self.d_generator_summary.add_scalar("d_generator_loss/g_loss_v2", self.d_g_loss_v2, global_step=0)
        self.d_generator_summary.add_scalar("d_generator_loss/d_gen_loss_fm", self.d_gen_loss_fm, global_step=0)
        self.d_generator_summary.add_scalar("d_generator_loss/d_gen_loss_adv", self.d_gen_loss_adv, global_step=0)
        self.d_generator_summary.add_scalar("d_generator_loss/d_generator", self.d_gen_loss, global_step=0)
        self.d_generator_summary.close()"""

    def gen_input_noise(self, num_sample, T, noise_dim):
        return np.random.uniform(size=[num_sample, T, noise_dim])
    
    def train(self):

        #  prepare training data for c
        continuous_x = self.c_data_sample[: int(0.9 * self.c_data_sample.shape[0]), :, :]
        continuous_x_test = self.c_data_sample[int(0.9 * self.c_data_sample.shape[0]) : , :, :]

        # prepare training data for d
        discrete_x = self.d_data_sample[: int(0.9 * self.d_data_sample.shape[0]), :, :]
        discrete_x_test = self.d_data_sample[int(0.9 * self.d_data_sample.shape[0]):, :, :]

        # num of batches
        data_size = continuous_x.shape[0]
        num_batches = data_size // self.batch_size

        # pretrain step
        print('start pretraining')
        global_id = 0

        for pre in range(self.num_pre_epochs):

            # prepare data for training dataset (same index)
            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            '''if self.conditional:
                label_data_random = label_data[random_idx]'''

            # validation data
            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            print("pretraining epoch %d" % pre)

            c_real_data_lst = []
            c_rec_data_lst = []
            d_real_data_lst = []
            d_rec_data_lst = []

            for b in range(num_batches):

                feed_dict = {}
                # feed d data
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                # feed c data
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                # feed label
                '''if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]'''

                # Pretrain the discrete and continuous vae loss
                _ = self.sess.run(self.c_v_op_pre, feed_dict=feed_dict)
                if ((pre + 1) % self.epoch_loss_freq == 0 or pre == self.num_pre_epochs - 1):
                    summary_result = self.sess.run(self.c_vae_summary, feed_dict=feed_dict)
                    #self.summary_writer.add_summary(summary_result, global_id)

                _ = self.sess.run(self.d_v_op_pre, feed_dict=feed_dict)
                if ((pre + 1) % self.epoch_loss_freq == 0 or pre == self.num_pre_epochs - 1):
                    summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                    #self.summary_writer.add_summary(summary_result, global_id)

                global_id += 1
            
                if ((pre + 1) % self.epoch_ckpt_freq == 0 or pre == self.num_pre_epochs - 1):
                    # real data vs. reconstructed data 
                    real_data, rec_data = self.sess.run([self.c_real_data_pl, self.c_decoded_output], feed_dict=feed_dict)
                    c_real_data_lst.append(real_data)
                    c_rec_data_lst.append(rec_data)

                    # real data vs. reconstructed data (rounding to 0 or 1)
                    real_data, rec_data = self.sess.run([self.d_real_data_pl, self.d_decoded_output], feed_dict=feed_dict)
                    d_real_data_lst.append(real_data)
                    d_rec_data_lst.append(np_rounding(rec_data))

            # visualize
            if ((pre + 1) % self.epoch_ckpt_freq == 0 or pre == self.num_pre_epochs - 1):
                visualise_vae(continuous_x_random, np.vstack(c_rec_data_lst), discrete_x_random, np.vstack(d_rec_data_lst), inx=(pre+1))
                print("finish vae reconstructed data saving in pre-epoch " + str(pre))

        np.savez('data/fake/vae.npz', c_real=np.vstack(c_real_data_lst), c_rec=np.vstack(c_rec_data_lst),
                 d_real=np.vstack(d_real_data_lst), d_rec=np.vstack(d_rec_data_lst))
        
        # saving the pretrain model 
        save_path = os.path.join(self.checkpoint_dir, "pretrain_vae_{}".format(global_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save(global_id=global_id -1, model_name= "m3gan", checkpoint_dir=save_path)
        print("finish the pretraining")

        # start jointly training -------------------
        print("start join training")

        for e in range(self.num_epochs):
            # prepare data for training dataset (same index)
            random_idx = np.random.permutation(data_size)
            continuous_x_random = continuous_x[random_idx]
            discrete_x_random = discrete_x[random_idx]
            '''if self.conditional:
                label_data_random = label_data[random_idx]'''
            
            # validation data
            random_idx_ = np.random.permutation(continuous_x_test.shape[0])
            continuous_x_test_batch = continuous_x_test[random_idx_][:self.batch_size, :, :]
            discrete_x_test_batch = discrete_x_test[random_idx_][:self.batch_size, :, :]

            print("training epoch %d" % e)

            for b in range(num_batches):
                feed_dict = {}
                # feed c
                feed_dict[self.c_real_data_pl] = continuous_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.c_gen_input_noise_pl] = self.gen_input_noise(self.batch_size, self.time_steps, noise_dim=self.c_noise_dim)
                feed_dict[self.c_vae_test_data_pl] = continuous_x_test_batch
                # feed d
                feed_dict[self.d_real_data_pl] = discrete_x_random[b * self.batch_size: (b + 1) * self.batch_size]
                feed_dict[self.d_gen_input_noise_pl] = self.gen_input_noise(self.batch_size, self.time_steps, noise_dim=self.d_noise_dim)
                feed_dict[self.d_vae_test_data_pl] = discrete_x_test_batch
                # if conditional, feed label
                '''if self.conditional:
                    feed_dict[self.real_data_label_pl] = label_data_random[b * self.batch_size: (b + 1) * self.batch_size]'''
                
                # training d
                for _ in range(self.d_rounds):
                    #_, d_summary_result = self.sess.run([self.d_d_op, self.d_discriminator_summary], feed_dict=feed_dict)
                    _ = self.sess.run(self.d_d_op, feed_dict=feed_dict)
                    if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        d_summary_result = self.sess.run(self.d_discriminator_summary, feed_dict=feed_dict)
                        self.summary_writer.add_summary(d_summary_result, global_id)

                    _ = self.sess.run(self.c_d_op, feed_dict=feed_dict)
                    if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        c_summary_result = self.sess.run(self.c_discriminator_summary, feed_dict=feed_dict)
                        self.summary_writer.add_summary(c_summary_result, global_id)

                 # training g
                for _ in range(self.g_rounds):
                    _ = self.sess.run(self.d_g_op, feed_dict=feed_dict)
                    if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        d_summary_result = self.sess.run(self.d_generator_summary, feed_dict=feed_dict)
                        self.summary_writer.add_summary(d_summary_result, global_id)

                    _ = self.sess.run(self.c_g_op, feed_dict=feed_dict)
                    if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        c_summary_result = self.sess.run(self.c_generator_summary, feed_dict=feed_dict)
                        self.summary_writer.add_summary(c_summary_result, global_id)

                # training v
                for _ in range(self.v_rounds):
                    _ = self.sess.run(self.d_v_op, feed_dict=feed_dict)
                    if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary_result, global_id)

                    _ = self.sess.run(self.c_v_op, feed_dict=feed_dict)
                    if ((e + 1) % self.epoch_loss_freq == 0 or e == self.num_epochs - 1):
                        summary_result = self.sess.run(self.d_vae_summary, feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary_result, global_id)

                global_id += 1

            if ((e + 1) % self.epoch_ckpt_freq == 0 or e == self.num_epochs - 1):
                data_gen_path = os.path.join("data/fake/", "epoch{}".format(e))
                if not os.path.exists(data_gen_path):
                    os.makedirs(data_gen_path)
                if self.conditional:
                    d_gen_data, c_gen_data = self.generate_data(num_sample=self.c_data_sample.shape[0], labels=self.statics_label)
                else:
                    d_gen_data, c_gen_data = self.generate_data(num_sample=self.c_data_sample.shape[0])
                np.savez(os.path.join(data_gen_path, "gen_data.npz"), c_gen_data=c_gen_data, d_gen_data=d_gen_data)
                visualise_gan(continuous_x_random, c_gen_data, discrete_x_random, d_gen_data, inx=(e+1))
                print('finish generated data saving in epoch ' + str(e))

    def generate_data(self, num_sample, labels=None):
        d_gen_data = []
        c_gen_data = []
        round_ = num_sample // self.batch_size

        for i in range(round_):
            d_gen_input_noise = self.gen_input_noise(
                num_sample=self.batch_size, T=self.time_steps, noise_dim=self.d_noise_dim)
            c_gen_input_noise = self.gen_input_noise(
                num_sample=self.batch_size, T=self.time_steps, noise_dim=self.c_noise_dim)
            
            feed_dict = {}
            feed_dict[self.d_gen_input_noise_pl] = d_gen_input_noise
            feed_dict[self.c_gen_input_noise_pl] = c_gen_input_noise
            if self.conditional:
                feed_dict[self.real_data_label_pl] = \
                    labels[i * self.batch_size: (i + 1) * self.batch_size]
                
            d_gen_data_, c_gen_data_ = self.sess.run([self.d_gen_decoded, self.c_gen_decoded], feed_dict=feed_dict)
            d_gen_data.append(np.stack(d_gen_data_, axis=1))
            c_gen_data.append(np.stack(c_gen_data_, axis=1))

        d_gen_data = np.concatenate(d_gen_data, axis=0)
        c_gen_data = np.concatenate(c_gen_data, axis=0)

        return np_rounding(d_gen_data), c_gen_data