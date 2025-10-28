# generator/scripts/check_dataset.py

import pandas as pd
from rdkit import Chem

# Load the SMILES dataset
df = pd.read_csv('C:\molvista-ai\generator\data\zinc_subset.csv')

valid_count = 0
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        valid_count += 1

print(f"Total SMILES: {len(df)}")
print(f"Valid SMILES: {valid_count}")
