import pandas as pd

df = pd.read_csv("C:\molvista-ai\generator\data\moses_train.csv")  # Or path to your file
print("Columns in CSV:", df.columns.tolist())
print(df.head())
