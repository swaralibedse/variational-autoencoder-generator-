import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)  # h: (1, batch, hidden)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, max_len):
        super(Decoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, z, teacher_inputs=None):
        batch_size = z.size(0)
        h = self.latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden)
        if teacher_inputs is not None:
            emb = self.embedding(teacher_inputs)
            output, _ = self.gru(emb, h)
            logits = self.fc_out(output)
        else:
            outputs = []
            input_token = torch.full((batch_size, 1), 1).to(z.device)  # [START] = 1
            for _ in range(self.max_len):
                emb = self.embedding(input_token)
                output, h = self.gru(emb, h)
                logits = self.fc_out(output.squeeze(1))
                outputs.append(logits.unsqueeze(1))
                input_token = logits.argmax(dim=1, keepdim=True)
            logits = torch.cat(outputs, dim=1)
        return logits

class MoleculeVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, max_len):
        super(MoleculeVAE, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, latent_dim, max_len)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.decoder(z, teacher_inputs=x)
        return logits, mu, logvar
