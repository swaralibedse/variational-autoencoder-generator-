import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from generator.tokenizer import SmilesTokenizer
from generator.vae import MoleculeVAE

# ========== Load MOSES dataset ==========
df = pd.read_csv("generator/data/moses_train.csv")
smiles_list = df["SMILES"].dropna().unique().tolist()

# ========== Tokenize ==========
tokenizer = SmilesTokenizer(smiles_list)
tokenized = [tokenizer.encode(s, max_len=120) for s in smiles_list]
input_tensor = pad_sequence([torch.tensor(t) for t in tokenized], batch_first=True, padding_value=tokenizer.pad_token_idx)

# ========== Prepare DataLoader ==========
dataset = TensorDataset(input_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# ========== Initialize Model ==========
vocab_size = tokenizer.vocab_size
pad_idx = tokenizer.pad_token_idx
model = MoleculeVAE(vocab_size=vocab_size, pad_idx=pad_idx)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")

# ========== Resume Support ==========
start_epoch = 0
model_dir = "generator/models"
os.makedirs(model_dir, exist_ok=True)

existing_epochs = [
    int(f.split("vae_model_epoch")[1].split(".pt")[0])
    for f in os.listdir(model_dir)
    if f.startswith("vae_model_epoch") and f.endswith(".pt")
]

if existing_epochs:
    latest_epoch = max(existing_epochs)
    checkpoint_path = f"{model_dir}/vae_model_epoch{latest_epoch}.pt"
    print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    start_epoch = latest_epoch + 1
else:
    print("🆕 No checkpoint found. Starting from scratch.")

# ========== Training Loop ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
losses = []

for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0.0
    recon_loss_total = 0.0
    kl_loss_total = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        x = batch[0].to(device)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)

        # Flatten for loss calculation
        x_hat = x_hat.view(-1, x_hat.size(-1))
        x = x.view(-1)

        # Reconstruction loss
        recon_loss = criterion(x_hat, x)

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()

    avg_epoch_loss = epoch_loss / len(dataset)
    print(f"✅ Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")
    losses.append((recon_loss_total / len(dataset), kl_loss_total / len(dataset)))

    # Save after each epoch
    torch.save(model.state_dict(), f"{model_dir}/vae_model_epoch{epoch}.pt")

# ========== Save Final Model ==========
torch.save(model.state_dict(), "C:\molvista-ai\generator\models/vae_model.pt")


# ========== Plot and Save Loss Curve ==========
recon_losses, kl_losses = zip(*losses)
plt.plot(recon_losses, label="Reconstruction Loss")
plt.plot(kl_losses, label="KL Divergence Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("generator/loss_curve.png")
print("📈 Saved loss curve to generator/loss_curve.png")
