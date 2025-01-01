import pickle
import pandas as pd


# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

X_new = pd.DataFrame({'X': [5]})
y_pred = model.predict(X_new)

print(f"Prediction for X = {X_new.values.flatten()[0]}: {y_pred[0]}")
