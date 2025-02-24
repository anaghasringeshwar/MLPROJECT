import pandas as pd
import os

# Define paths
data_path = "/kaggle/input/fashion-product-images-dataset"  # Replace this with the actual path to your dataset
dataset_name = "fashion-product-images-dataset"  # Replace with your dataset name
dataset_path = os.path.join(data_path, dataset_name)
fashion_dataset_path = os.path.join(dataset_path, 'fashion-dataset')

# Load the styles CSV file, skipping bad lines
styles_csv_path = os.path.join(fashion_dataset_path, 'styles.csv')
styles_df = pd.read_csv(styles_csv_path, on_bad_lines='skip')

# Print the first few rows to verify
print("Styles DataFrame:")
print(styles_df.head())

# Optional: Check for problematic lines
with open(styles_csv_path, 'r') as file:
    lines = file.readlines()
    print("Problematic line (6044):", lines[6043])  # line index is zero-based
