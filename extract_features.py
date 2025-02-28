import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm
import pickle

# Load the dataset
df = pd.read_csv("styles_1000.csv")  # Change if filename differs

# Path to your images folder
IMAGE_FOLDER = "path_to_your_images_folder"  # UPDATE THIS

# Load the pre-trained VGG16 model (without classification layers)
base_model = VGG16(weights="imagenet", include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract image features
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess for VGG16

        features = model.predict(img_array)
        features = features.flatten()  # Convert to 1D vector
        return features
    except:
        return np.zeros((model.output_shape[-1],))  # Return zeros if error

# Extract features for all images
image_features = {}

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_id = row["id"]
    img_path = os.path.join(IMAGE_FOLDER, f"{img_id}.jpg")  # Update if needed

    if os.path.exists(img_path):
        image_features[img_id] = extract_features(img_path)
    else:
        image_features[img_id] = np.zeros((model.output_shape[-1],))  # Handle missing images

# Save extracted features
with open("image_features.pkl", "wb") as f:
    pickle.dump(image_features, f)

print("✅ Image features extracted and saved!")
