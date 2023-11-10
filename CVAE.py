import torch
import torch.nn as nn

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
        self.linear = nn.Linear(hidden_size, input_size)  # Corrected input size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out)
        return out

# Set random seed for reproducibility
torch.manual_seed(42)

# Define input parameters
input_size = 10
hidden_size = 5
num_layers = 3

# Generate random input data
batch_size = 5
sequence_length = 50
input_data = torch.randn(batch_size, sequence_length, input_size)

#print('input data is', input_data)

# Instantiate the encoder and decoder
encoder = EncoderLSTM(input_size, hidden_size, num_layers)
decoder = DecoderLSTM(input_size, hidden_size, num_layers)

# Forward pass through the encoder
encoded_data = encoder(input_data)

#print('encoded data is',encoded_data)

# Forward pass through the decoder
decoded_data = decoder(encoded_data)

#print('decoded data is',decoded_data)

# Print the shapes of the input, encoded output, and decoded output
print("Input shape:", input_data.shape)
print("Encoded output shape:", encoded_data.shape)
print("Decoded output shape:", decoded_data.shape)
