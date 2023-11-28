import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        return out

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()  # Add a sigmoid activation

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out)
        out = self.sigmoid(out)  # Apply sigmoid activation
        return out

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(VAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        encoded_data = self.encoder(x)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

# Set random seed for reproducibility
torch.manual_seed(42)

# Instantiate the VAE
vae = VAE(input_size=4, hidden_size=20, num_layers=3)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Use Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Generate random input data
batch_size = 5
sequence_length = 1000
input_data = torch.randn(batch_size, sequence_length, 4)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    output = vae(input_data)

    # Compute the loss
    loss = criterion(output, torch.sigmoid(input_data))  # Apply sigmoid to the target

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the trained VAE
test_input = torch.randn(1, sequence_length, 4)
decoded_output = vae(test_input)

print("\nOriginal Input:")
print(test_input.squeeze().detach().numpy())
print("\nDecoded Output:")
print(decoded_output.squeeze().detach().numpy())


print("end")
