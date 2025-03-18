import pickle

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Check if it's loaded
print("Model loaded successfully!")
print(model)  # This will print model details
