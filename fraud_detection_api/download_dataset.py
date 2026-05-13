
import kagglehub
import pandas as pd
import os

print("Downloading Credit Card Fraud Detection dataset...")

# Download the dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(f"Dataset downloaded to: {path}")

# Find and load the CSV file
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_path = os.path.join(path, file)
        break

# Load and verify
df = pd.read_csv(csv_path)
print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# Save to your data folder
df.to_csv('data/creditcard.csv', index=False)
print("\nDataset saved to: data/creditcard.csv")

# Quick verification
print(f"\nFraud cases: {df['Class'].sum()} out of {len(df)} ({df['Class'].mean()*100:.3f}%)")