import torch
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, pad_idx):
    recon_x = recon_x.permute(0, 2, 1)
    loss_recon = F.cross_entropy(recon_x, x, ignore_index=pad_idx)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_recon + kl_loss, loss_recon.item(), kl_loss.item()

def train_epoch(model, data_loader, optimizer, pad_idx, device):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    for batch in data_loader:
        x = batch.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(x)
        loss, recon, kl = loss_function(logits, x, mu, logvar, pad_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon
        total_kl += kl
    return total_loss / len(data_loader), total_recon / len(data_loader), total_kl / len(data_loader)
