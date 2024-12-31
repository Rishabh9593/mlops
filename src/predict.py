import pickle
import numpy as np

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input features
# If your model was trained with a single feature, for example, you need to make sure the input data
# for prediction also has only one feature.
# For demonstration, let's assume the model expects 1 feature
X_new = np.array([[5]])  # Example: Input with 1 feature (ensure this matches training data)

# If your model was trained with multiple features (e.g., 4 features), reshape the input accordingly
# X_new = np.array([[feature1, feature2, feature3, feature4]])  # Uncomment and replace with actual data

# Make a prediction
y_pred = model.predict(X_new)

# Print the prediction
print(f"Prediction: {y_pred}")
