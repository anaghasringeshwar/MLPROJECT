import pickle

# Load the trained model
with open("saved_model.pkl", "rb") as file:
    model = pickle.load(file)

# Test with an image_id (try changing this number)
test_image_id = 100  # Example image_id

# Make a prediction (some models require 2D input, so we wrap it in [[]])
try:
    recommendations = model.predict([[test_image_id]])
    print("Model Output:", recommendations)  # Debugging
except Exception as e:
    print("Model Error:", e)  # If there's an issue
