import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors  # Example model

# Load extracted features
with open("image_features.pkl", "rb") as f:
    data = pickle.load(f)

# Ensure dictionary keys are strings
data = {str(k): v for k, v in data.items()}

# Convert dictionary to feature matrix and item mapping
image_ids = list(data.keys())  # List of image names/IDs
feature_matrix = np.array(list(data.values()))  # Convert features to NumPy array

# Train a simple recommendation model (using Nearest Neighbors)
model = NearestNeighbors(n_neighbors=5, metric="cosine")
model.fit(feature_matrix)

# Store the trained model along with metadata
model_data = {
    "model": model,  # The trained recommendation model
    "image_ids": image_ids,  # List of image IDs for retrieval
    "features": feature_matrix,  # Feature matrix (optional)
}

# Save the trained model and metadata
with open("saved_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model trained and saved as saved_model.pkl")
