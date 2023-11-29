import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder = nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        mu = self.fc_mu(h_n[-1, :])
        logvar = self.fc_logvar(h_n[-1, :])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        output, _ = self.decoder(z.unsqueeze(1))
        output = self.fc_output(output)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Load the data
with open('continuous.pkl', 'rb') as f:
    my_data = pickle.load(f)

# Convert my_data to a PyTorch tensor
my_data = torch.tensor(my_data, dtype=torch.float32)

# Initialize the VAE model, optimizer, and other hyperparameters
input_size = 3
hidden_size = 10
latent_size = 5

vae = VAE(input_size, hidden_size, latent_size)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_divergence
    return total_loss

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    recon_batch, mu, logvar = vae(my_data)

    # Calculate the loss
    loss = loss_function(recon_batch, my_data, mu, logvar)

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Print the loss every 100 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

print("end!")
