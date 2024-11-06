import pandas as pd
import warnings
import csv
import numpy as np
import ast
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import KFold

#test commit

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 2)

# Load the dataset
df = pd.read_csv("Golden_list.csv")
df = df[["name", "legalities", "power", "toughness", "prices", "rarity", "set_id", "artist", "color_identity", "released_at", "keywords", "full_art"]]
df = df.head(1000)
df = df[["power", "toughness", "rarity", "prices"]]
df["power"] = df["power"].replace('*', np.nan)
df["power"] = df["power"].replace('1+*', np.nan)
df = df.dropna()

df[["power"]] = df[["power"]].astype(float)
df[["toughness"]] = df[["toughness"]].astype(float)

df["rarity"] = df["rarity"].replace({'uncommon': 1, 'common': 2, 'rare': 3, 'mythic': 4})
df["price"] = df["prices"].apply(lambda x: ast.literal_eval(x).get('eur'))
df["price"] = pd.to_numeric(df["price"], errors='coerce')
df = df.drop(columns=["prices"])
df = df.dropna(subset=["price"])  # Drop rows with missing prices

means = df.mean()
stds = df.std()

df2 = (df[means.index] - means) / stds  # Normalized dataset

# Prepare data for ANN
X = df2[["power", "toughness", "rarity"]].values
y = df2["price"].values
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the neural network model
class PricePredictionANN(nn.Module):
    def __init__(self, h):
        super(PricePredictionANN, self).__init__()
        self.fc1 = nn.Linear(3, h)
        self.fc2 = nn.Linear(h, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, epochs):
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val).item()
    
    return val_loss

# Two-level cross-validation
h_values = [1, 2, 5, 8, 10]  # Different numbers of hidden units to test
K1 = 10  # Outer cross-validation folds
K2 = 10  # Inner cross-validation folds

epochs = 500
learning_rate = 0.001

outer_kfold = KFold(n_splits=K1, shuffle=True, random_state=42)
generalization_errors = []
baseline_errors = []
results_table = []  # To store results for the table

for i, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(X), 1): #outer fold (loop)
    X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
    y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
    
    # Baseline model: predicting the mean of y_outer_train, no parameter training
    baseline_mean = y_outer_train.mean()
    baseline_predictions = torch.full((len(y_outer_test), 1), baseline_mean)  # Predicting mean for all test samples
    baseline_loss = nn.MSELoss()(baseline_predictions, y_outer_test).item()
    baseline_errors.append(baseline_loss)
    
    inner_kfold = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    # To store validation errors for each model with different h
    model_validation_errors = {h: [] for h in h_values}
    
    for inner_train_idx, inner_val_idx in inner_kfold.split(X_outer_train):
        X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
        y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
        
        # Train models for each h value
        for h in h_values:
            model = PricePredictionANN(h)
            val_loss = train_and_evaluate_model(model, X_inner_train, y_inner_train, X_inner_val, y_inner_val, epochs)
            model_validation_errors[h].append(val_loss)
    
    # Compute average validation error for each h
    avg_validation_errors = {h: np.mean(model_validation_errors[h]) for h in h_values}
    
    # Select the best model (with lowest validation error)
    best_h = min(avg_validation_errors, key=avg_validation_errors.get)
    print(f"Best h in this outer fold {i}: {best_h}")
    
    # Train the best model on the entire outer training set
    best_model = PricePredictionANN(best_h)
    train_and_evaluate_model(best_model, X_outer_train, y_outer_train, X_outer_train, y_outer_train, epochs)
    
    # Evaluate the best model on the outer test set
    best_model.eval()
    with torch.no_grad():
        test_predictions = best_model(X_outer_test)
        test_loss = nn.MSELoss()(test_predictions, y_outer_test).item()
    
    generalization_errors.append(test_loss)
    print(f"Test loss in this outer fold {i}: {test_loss}")
    
    # Store the results for the table
    results_table.append([i, best_h, test_loss, baseline_loss])

# Estimate of the generalization error
#estimated_generalization_error = np.mean(generalization_errors)
#estimated_baseline_error = np.mean(baseline_errors)
#print(f"Estimated Generalization Error: {estimated_generalization_error}")
#print(f"Estimated Baseline Error: {estimated_baseline_error}")

# Display results in a table
results_df = pd.DataFrame(results_table, columns=["Outer Fold", "Best h", "E_test (Neural Net)", "E_test (Baseline)"])
print("\nResults Table:\n")
print(results_df)

# Optionally save the results to a CSV file
results_df.to_csv('results_table.csv', index=False)
