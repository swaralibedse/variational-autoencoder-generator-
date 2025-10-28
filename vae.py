import torch
import torch.nn as nn

class MoleculeVAE(nn.Module):
    def __init__(self, vocab_size, pad_idx, hidden_size=256, latent_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.encoder(x)
        h = h[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        dec_input = self.decoder_input(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_output, _ = self.decoder(dec_input)
        logits = self.output(dec_output)
        return logits, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
