import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load extracted features
with open("image_features.pkl", "rb") as f:
    data = pickle.load(f)

# Convert dictionary keys (image IDs) to strings for consistency
data = {str(k): v for k, v in data.items()}

# Convert dictionary to arrays
image_ids = list(data.keys())  # Ensure all IDs are strings
feature_matrix = np.array(list(data.values()))

# Path to images folder
IMAGE_FOLDER = "images"  # Ensure this is correct

# Function to find similar images
def find_similar(image_id, top_n=5):
    image_id = str(image_id)  # Convert to string for consistency

    if image_id not in data:
        print(f"❌ Image ID {image_id} not found!")
        return []

    query_vector = data[image_id].reshape(1, -1)  # Reshape for similarity comparison
    similarities = cosine_similarity(query_vector, feature_matrix)[0]
    
    # Get top N similar images (excluding the input image itself)
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]
    similar_image_ids = [image_ids[i] for i in similar_indices]
    
    return similar_image_ids

# Function to display images
def show_images(query_id, similar_ids):
    fig, axes = plt.subplots(1, len(similar_ids) + 1, figsize=(15, 5))

    # Load and show query image
    query_img_path = os.path.join(IMAGE_FOLDER, f"{query_id}.jpg")
    
    if not os.path.exists(query_img_path):
        print(f"❌ Query image {query_id}.jpg not found!")
        return
    
    query_img = cv2.imread(query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    # Load and show similar images
    for i, sim_id in enumerate(similar_ids):
        sim_img_path = os.path.join(IMAGE_FOLDER, f"{sim_id}.jpg")
        
        if not os.path.exists(sim_img_path):
            print(f"⚠️ Skipping {sim_id}.jpg (Image not found)")
            continue  # Skip missing images
        
        sim_img = cv2.imread(sim_img_path)
        sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
        axes[i + 1].imshow(sim_img)
        axes[i + 1].set_title(f"Similar {i+1}")
        axes[i + 1].axis("off")

    plt.show()

if __name__ == "__main__":
    while True:
        test_image_id = input("\nEnter an Image ID (or type 'exit' to quit): ").strip()

        if test_image_id.lower() == "exit":
            print("Exiting program. Goodbye! 👋")
            break

        similar_images = find_similar(test_image_id)

        if similar_images:
            show_images(test_image_id, similar_images)
