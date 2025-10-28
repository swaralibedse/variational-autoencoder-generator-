import torch
import pandas as pd
from generator.tokenizer import SmilesTokenizer
from generator.vae import MoleculeVAE
import random

# Load data + tokenizer
df = pd.read_csv("generator/data/moses_train.csv")
smiles_list = df["SMILES"].dropna().unique().tolist()
tokenizer = SmilesTokenizer(smiles_list)

model = MoleculeVAE(vocab_size=tokenizer.vocab_size, pad_idx=tokenizer.pad_token_idx)
model.load_state_dict(torch.load("generator/models/vae_model.pt", map_location="cpu"))
model.eval()

def sample_model(model, tokenizer, num_samples=100):
    generated_smiles = []
    for _ in range(num_samples):
        z = torch.randn(1, model.latent_size)
        with torch.no_grad():
            logits = model.decode(z, max_len=120)
            token_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
            smiles = tokenizer.decode(token_ids)
            generated_smiles.append(smiles)
    return generated_smiles

generated = sample_model(model, tokenizer, num_samples=100)
unique = len(set(generated))
print(f"Generated: {len(generated)} SMILES")
print(f"Unique: {unique} ({unique / len(generated) * 100:.2f}%)")
