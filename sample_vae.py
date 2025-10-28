import torch
import pandas as pd
from generator.tokenizer import SmilesTokenizer
from generator.vae import MoleculeVAE

# Load dataset and tokenizer
df = pd.read_csv("generator/data/moses_train.csv")
smiles_list = df["SMILES"].dropna().unique().tolist()
tokenizer = SmilesTokenizer(smiles_list)

# Initialize model
model = MoleculeVAE(vocab_size=tokenizer.vocab_size, pad_idx=tokenizer.pad_token_idx)
model.load_state_dict(torch.load("generator/models/vae_model.pt", map_location="cpu"))
model.eval()

# Select sample SMILES
sample_smiles = smiles_list[0]
encoded = tokenizer.encode(sample_smiles, max_len=120)
input_tensor = torch.tensor(encoded).unsqueeze(0)

# Generate output
with torch.no_grad():
    output, _, _ = model(input_tensor)
    decoded = torch.argmax(output, dim=-1).squeeze().tolist()

print("✅ Sample Input vs Output")
print("Original:", sample_smiles)
print("Decoded :", tokenizer.decode(decoded))
