import pickle

# Load the saved pickle file
with open("saved_model.pkl", "rb") as file:
    model_data = pickle.load(file)

print("Type of saved_model.pkl:", type(model_data))

# If it's a dictionary, print the keys
if isinstance(model_data, dict):
    print("Keys inside pickle file:", model_data.keys())
