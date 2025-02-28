import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model (image features)
with open("../saved_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to find similar items
def find_similar(image_feature, top_n=5):
    similarities = cosine_similarity(image_feature.reshape(1, -1), model["features"])[0]
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]  # Exclude input image
    similar_image_ids = [model["image_ids"][i] for i in similar_indices]
    return similar_image_ids
