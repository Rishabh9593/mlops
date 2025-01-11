import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

# Sample dataset
df = pd.DataFrame({'X': range(10), 'y': [2*x for x in range(10)]})
X = df[['X']]
y = df['y']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

# Save model
with open("models/model.pkl", 'wb') as f:
    pickle.dump(model, f)
