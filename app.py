from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 🟢 Load the trained model
with open("saved_model.pkl", "rb") as f:
    model_data = pickle.load(f)
image_ids = model_data["image_ids"]  # List of image IDs
feature_matrix = np.array(model_data["features"])  # Feature vectors
metadata = pd.DataFrame.from_dict(model_data["metadata"])  # Image metadata
COMPLEMENTARY_CATEGORIES = model_data["complementary_categories"]  # Complementary mappings

IMAGE_FOLDER = "images"  # Ensure images are stored in this folder

# 🔹 Track current image index for swiping
current_index = 0
@app.route("/")
def frontpage():
    """ Render the frontpage first """
    return render_template("frontpage.html")




@app.route("/index")
def index():
    """ Render the main swiper page """
    return render_template("index.html")

@app.route("/get_outfit", methods=["GET"])
def get_outfit():
    """ Serve the next image for swiping left/right """
    global current_index
    if current_index >= len(image_ids):
        current_index = 0  # Restart if at end

    image_id = image_ids[current_index]
    current_index += 1  # Move to next image

    return jsonify({"image": f"/images/{image_id}.jpg"})  # Serve image URL

def find_similar(image_id, top_n=5):
    """Find complementary outfit recommendations."""
    image_id = str(image_id)  # Ensure it's a string

    if image_id not in image_ids:
        return []

    # Get query image details
    if image_id not in metadata.index:
        return []

    query_vector = feature_matrix[image_ids.index(image_id)].reshape(1, -1)
    query_gender = metadata.loc[image_id, "gender"]
    query_category = metadata.loc[image_id, "subCategory"]

    # Get valid image IDs with same gender
    valid_images = metadata[metadata["gender"] == query_gender].index.tolist()

    # Prefer complementary categories
    preferred_categories = COMPLEMENTARY_CATEGORIES.get(query_category, [])
    preferred_images = metadata[metadata["subCategory"].isin(preferred_categories)].index.tolist()

    # Compute similarity scores (contrast)
    valid_indices = [image_ids.index(img) for img in valid_images if img in image_ids]
    valid_features = feature_matrix[valid_indices]
    similarities = cosine_similarity(query_vector, valid_features)[0]
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order

    # Select the top-N results
    similar_images = [valid_images[i] for i in sorted_indices]
    final_results = [img for img in similar_images if img in preferred_images][:top_n]

    # If not enough preferred matches, fill with other same-gender items
    if len(final_results) < top_n:
        remaining = [img for img in similar_images if img not in final_results][:top_n - len(final_results)]
        final_results.extend(remaining)

    return final_results

@app.route("/get_recommendations", methods=["POST"])
def get_recommendations():
    """ Fetch recommended outfits based on user selection """
    data = request.json
    selected_image_id = data.get("image_id")

    recommendations = find_similar(selected_image_id)
    recommended_paths = [f"/images/{img}.jpg" for img in recommendations]

    return jsonify({"recommendations": recommended_paths})

@app.route("/images/<filename>")
def serve_image(filename):
    """ Serve images dynamically """
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/swiper')
def swiper():
    return render_template('index.html')  # Ensure 'index.html' exists in /templates


@app.route("/suggestions")
def suggestions():
    """ Render the suggestions page """
    return render_template("suggestions.html")

if __name__ == "__main__":
    app.run(debug=True)
