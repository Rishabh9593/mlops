import pickle
import numpy as np
import sys

def load_model(model_path):
    """Load the trained model from the specified path."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_prediction(model, input_data):
    """Use the trained model to make predictions."""
    prediction = model.predict(input_data)
    return prediction

def main():
    """Main function to load the model and make predictions."""
    # Path to the model file
    model_path = 'models/model.pkl'

    # Sample input for prediction (adjust this based on your model input)
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example for Iris dataset, modify as needed

    # Load the model
    model = load_model(model_path)

    # Make a prediction
    prediction = make_prediction(model, sample_input)

    # Print the prediction result
    print(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
