import pickle
import numpy as np

# Load extracted features
with open("image_features.pkl", "rb") as f:
    data = pickle.load(f)

# Check if data is loaded correctly
if not data:
    print("Error: features.pkl is empty or not loaded properly.")
    exit()

# Check the total number of images processed
print(f"✅ Total images with extracted features: {len(data)}")

# Print a sample of extracted features
for i, (image_id, features) in enumerate(data.items()):
    print(f"🖼️ Image ID: {image_id}, Feature Shape: {features.shape}")
    if i == 4:  # Show only first 5 samples
        break
