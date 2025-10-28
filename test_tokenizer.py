import sys
import os

# Add the generator directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer import SmilesTokenizer  
import pandas as pd

# Load dataset
df = pd.read_csv('C:\molvista-ai\generator\data\generated.csv')
smiles_list = df['SMILES'].tolist()

# Initialize tokenizer
tokenizer = SmilesTokenizer(smiles_list)
print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")

# Test one sample
example = smiles_list[0]
encoded = tokenizer.encode(example, max_len=64)
decoded = tokenizer.decode(encoded)

print(f"Original: {example}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
