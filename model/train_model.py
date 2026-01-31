import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("../data/expense_data.csv")

# Select input (X) and output (Y)
X = data[['Monthly_Income']]
y = data['Monthly_Expense']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Save trained model
with open("expense_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained successfully and saved as expense_model.pkl")
