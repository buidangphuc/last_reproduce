import torch
import torch.nn as nn

class LSTM_VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(LSTM_VAE, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Variables
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder Part
        self.encoder_lstm = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    batch_first=True,
                                    num_layers=num_layers)
        self.mean = nn.Linear(hidden_size * num_layers, latent_size)
        self.log_variance = nn.Linear(hidden_size * num_layers, latent_size)

        # Decoder Part
        self.init_hidden_decoder = nn.Linear(latent_size, hidden_size * num_layers)
        self.decoder_lstm = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    batch_first=True,
                                    num_layers=num_layers)
        self.output = nn.Linear(hidden_size * num_layers, input_size)
        # Không sử dụng LogSoftmax vì đầu ra là liên tục

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states to zeros
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (hidden, cell)

    def encoder(self, sequences, hidden):
        # Pass the sequences through LSTM
        output_encoder, (h_n, c_n) = self.encoder_lstm(sequences, hidden)
        # h_n: (num_layers, batch, hidden_size)

        # Concatenate the final hidden states from all layers
        h_n_cat = h_n.permute(1, 0, 2).contiguous().view(h_n.size(1), -1)  # (batch, num_layers * hidden_size)

        # Compute mean and log variance
        mean = self.mean(h_n_cat)         # (batch, latent_size)
        log_var = self.log_variance(h_n_cat)  # (batch, latent_size)
        std = torch.exp(0.5 * log_var)

        # Reparameterization trick
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z, mean, log_var, (h_n, c_n)

    def decoder(self, z, hidden=None):
        # Initialize decoder hidden state from z
        hidden_decoder = self.init_hidden_decoder(z)  # (batch, hidden_size * num_layers)
        hidden_decoder = hidden_decoder.view(-1, self.num_layers, self.hidden_size)  # (batch, num_layers, hidden_size)
        hidden_decoder = hidden_decoder.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_size)
        cell_decoder = torch.zeros_like(hidden_decoder).to(self.device)  # Initialize cell state to zeros
        state_decoder = (hidden_decoder, cell_decoder)  # (h_0, c_0)

        # Initialize decoder input as zeros (start token can be modified if needed)
        batch_size = z.size(0)
        decoder_input = torch.zeros(batch_size, 1, z.size(1)).to(self.device)  # (batch, 1, input_size)

        # To store all outputs
        outputs = []

        # Generate sequence step-by-step
        for _ in range(100):  # Bạn có thể thay đổi số bước theo nhu cầu
            output, state_decoder = self.decoder_lstm(decoder_input, state_decoder)  # output: (batch, 1, hidden_size)
            output = output.contiguous().view(-1, self.hidden_size * self.num_layers)  # (batch, hidden_size * num_layers)
            x_hat = self.output(output)  # (batch, input_size)
            outputs.append(x_hat.unsqueeze(1))  # (batch, 1, input_size)
            decoder_input = x_hat.unsqueeze(1)  # (batch, 1, input_size)

        # Concatenate all outputs
        x_hat = torch.cat(outputs, dim=1)  # (batch, seq_len, input_size)

        return x_hat

    def forward(self, sequences, seq_lengths):
        """
        Forward pass through the VAE.

        Args:
            sequences (torch.Tensor): Input sequences of shape (1, seq_len, input_size).
            seq_lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size=1).

        Returns:
            x_hat (torch.Tensor): Reconstructed sequences of shape (1, seq_len, input_size).
            mean (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log variance of the latent distribution.
        """
        batch_size = sequences.size(0)

        # Initialize hidden state for encoder
        hidden = self.init_hidden(batch_size)

        # Encoder
        z, mean, log_var, hidden_enc = self.encoder(sequences, hidden)

        # Decoder
        x_hat = self.decoder(z)

        return x_hat, mean, log_var

    def generate(self, z, max_seq_len=100):
        """
        Generates a trajectory from a given latent vector z.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_size).
            max_seq_len (int): Maximum sequence length for the generated trajectory.

        Returns:
            torch.Tensor: Generated trajectory of shape (batch_size, max_seq_len, input_size).
        """
        self.eval()
        with torch.no_grad():
            batch_size = z.size(0)
            # Initialize decoder hidden state from z
            hidden_decoder = self.init_hidden_decoder(z)  # (batch, hidden_size * num_layers)
            hidden_decoder = hidden_decoder.view(-1, self.num_layers, self.hidden_size)  # (batch, num_layers, hidden_size)
            hidden_decoder = hidden_decoder.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_size)
            cell_decoder = torch.zeros_like(hidden_decoder).to(self.device)  # Initialize cell state to zeros
            state_decoder = (hidden_decoder, cell_decoder)  # (h_0, c_0)

            # Initialize decoder input as zeros
            decoder_input = torch.zeros(batch_size, 1, 2).to(self.device)  # Assuming input_size=2 (lat, lon)

            # To store all outputs
            outputs = []

            for _ in range(max_seq_len):
                output, state_decoder = self.decoder_lstm(decoder_input, state_decoder)  # output: (batch, 1, hidden_size)
                output = output.contiguous().view(-1, self.hidden_size * self.num_layers)  # (batch, hidden_size * num_layers)
                x_hat = self.output(output)  # (batch, input_size)
                outputs.append(x_hat.unsqueeze(1))  # (batch, 1, input_size)
                decoder_input = x_hat.unsqueeze(1)  # (batch, 1, input_size)

            # Concatenate all outputs
            x_hat = torch.cat(outputs, dim=1)  # (batch, max_seq_len, input_size)

            return x_hat

    def inference(self, z, max_seq_len=100):
        """
        Generates a trajectory from a given latent vector z step-by-step.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_size).
            max_seq_len (int): Maximum sequence length for the generated trajectory.

        Returns:
            torch.Tensor: Generated trajectory of shape (batch_size, max_seq_len, input_size).
        """
        self.eval()
        with torch.no_grad():
            batch_size = z.size(0)
            input_step = torch.zeros(batch_size, 1, 2).to(self.device)  # Assuming input_size=2 (lat, lon)

            # Initialize decoder hidden state from z
            hidden_decoder = self.init_hidden_decoder(z)  # (batch, hidden_size * num_layers)
            hidden_decoder = hidden_decoder.view(-1, self.num_layers, self.hidden_size)  # (batch, num_layers, hidden_size)
            hidden_decoder = hidden_decoder.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_size)
            cell_decoder = torch.zeros_like(hidden_decoder).to(self.device)
            state_decoder = (hidden_decoder, cell_decoder)

            generated = []
            for _ in range(max_seq_len):
                output, state_decoder = self.decoder_lstm(input_step, state_decoder)  # output: (batch, 1, hidden_size)
                output = output.contiguous().view(-1, self.hidden_size * self.num_layers)  # (batch, hidden_size * num_layers)
                x_hat = self.output(output)  # (batch, input_size)
                generated.append(x_hat.unsqueeze(1))  # (batch, 1, input_size)
                input_step = x_hat.unsqueeze(1)  # (batch, 1, input_size)

            generated = torch.cat(generated, dim=1)  # (batch, max_seq_len, input_size)
            return generated
