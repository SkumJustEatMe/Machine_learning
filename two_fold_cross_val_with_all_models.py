import pandas as pd
import warnings
import numpy as np
import ast
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.optim import Adam

from scipy.stats import ttest_rel

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

## loading dataset and preproccesing  ------------------------------------------------------------------------------------------------------------------------------------
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
df = df.dropna(subset=["price"])  

means = df.mean()
stds = df.std()

df2 = (df[means.index] - means) / stds  # Normalized dataset

#for ANN
X = df2[["power", "toughness", "rarity"]].values
y = df2["price"].values
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

## define neural network ------------------------------------------------------------------------------------------------------------------------------------
class PricePredictionANN(nn.Module):
    def __init__(self, h):
        super(PricePredictionANN, self).__init__()
        self.fc1 = nn.Linear(3, h)
        self.fc2 = nn.Linear(h, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##define helper functions for loop 
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
lambdas = [10**(-4), 10**(-2), 1, 10, 100]  # Different lambda values for Ridge Regression
K1 = 10  # Outer cross-validation folds
K2 = 10  # Inner cross-validation folds

epochs = 500
learning_rate = 0.001

outer_kfold = KFold(n_splits=K1, shuffle=True, random_state=42)
results_table = []

for i, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(X), 1):  # outer fold (loop)
    X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
    y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
    
    # Baseline model: predicting the mean of y_outer_train, no parameter training
    baseline_mean = y_outer_train.mean()
    baseline_predictions = np.full((len(y_outer_test), 1), baseline_mean)  # Predicting mean for all test samples
    baseline_loss = np.mean((baseline_predictions.flatten() - y_outer_test) ** 2)

    inner_kfold = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    # To store validation errors for each model with different h and lambda
    model_validation_errors = {h: [] for h in h_values}
    ridge_validation_errors = {lamb: [] for lamb in lambdas}

    for inner_train_idx, inner_val_idx in inner_kfold.split(X_outer_train):
        X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
        y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
        
        # Train models for each h value (Neural Network)
        for h in h_values:
            model = PricePredictionANN(h)
            val_loss = train_and_evaluate_model(model, torch.tensor(X_inner_train, dtype=torch.float32), 
                                                 torch.tensor(y_inner_train, dtype=torch.float32).view(-1, 1), 
                                                 torch.tensor(X_inner_val, dtype=torch.float32), 
                                                 torch.tensor(y_inner_val, dtype=torch.float32).view(-1, 1), 
                                                 epochs)
            model_validation_errors[h].append(val_loss)

        # Train Ridge Regression for each lambda
        for lamb in lambdas:
            ridge_model = Ridge(alpha=lamb)
            ridge_model.fit(X_inner_train, y_inner_train)
            val_loss_ridge = np.mean((ridge_model.predict(X_inner_val) - y_inner_val) ** 2)
            ridge_validation_errors[lamb].append(val_loss_ridge)

    # Compute average validation error for each model
    avg_validation_errors = {h: np.mean(model_validation_errors[h]) for h in h_values}
    avg_ridge_errors = {lamb: np.mean(ridge_validation_errors[lamb]) for lamb in lambdas}

    # Select the best model (with lowest validation error) for ANN
    best_h = min(avg_validation_errors, key=avg_validation_errors.get)
    
    # Train the best ANN model on the entire outer training set
    best_model = PricePredictionANN(best_h)
    train_and_evaluate_model(best_model, torch.tensor(X_outer_train, dtype=torch.float32), 
                              torch.tensor(y_outer_train, dtype=torch.float32).view(-1, 1), 
                              torch.tensor(X_outer_train, dtype=torch.float32), 
                              torch.tensor(y_outer_train, dtype=torch.float32).view(-1, 1), 
                              epochs)
    
    # Evaluate the best ANN model on the outer test set
    best_model.eval()
    with torch.no_grad():
        test_predictions = best_model(torch.tensor(X_outer_test, dtype=torch.float32))
        test_loss_ann = nn.MSELoss()(test_predictions, torch.tensor(y_outer_test, dtype=torch.float32).view(-1, 1)).item()

    # Evaluate Ridge Regression on the outer test set for each lambda and choose the best
    best_ridge_loss = float('inf')
    best_lambda = None
    for lamb in lambdas:
        ridge_model_final = Ridge(alpha=lamb)
        ridge_model_final.fit(X_outer_train, y_outer_train)
        test_loss_ridge = np.mean((ridge_model_final.predict(X_outer_test) - y_outer_test) ** 2)
        
        if test_loss_ridge < best_ridge_loss:
            best_ridge_loss = test_loss_ridge
            best_lambda = lamb

    print(f"Outer Fold {i}: Best h (ANN): {best_h}, Test loss (ANN): {test_loss_ann}, Best λ (Ridge): {best_lambda}, Test loss (Ridge): {best_ridge_loss}")

    # Store the results for the table
    results_table.append([i, best_h, test_loss_ann, baseline_loss, best_lambda, best_ridge_loss])

## convert results to df --------------------------------------------
results_df = pd.DataFrame(results_table, columns=["Outer Fold", "Best h (ANN)", "E_test (Neural Net)", "E_test (Baseline)", "Best λ (Ridge)", "E_test (Ridge)"])
print("\nResults Table:\n")
print(results_df)

results_df.to_csv('results_table.csv', index=False)

## statistical analysis 
# Extract generalization error columns
e_test_nn = results_df['E_test (Neural Net)']
e_test_baseline = results_df['E_test (Baseline)']
e_test_ridge = results_df['E_test (Ridge)']

# Perform paired t-tests
t_nn_baseline, p_nn_baseline = ttest_rel(e_test_nn, e_test_baseline)
t_nn_ridge, p_nn_ridge = ttest_rel(e_test_nn, e_test_ridge)
t_ridge_baseline, p_ridge_baseline = ttest_rel(e_test_ridge, e_test_baseline)

# Display the results
print("Paired t-test Results:")
print(f"Neural Net vs Baseline: t = {t_nn_baseline:.4f}, p = {p_nn_baseline:.4f}")
print(f"Neural Net vs Ridge: t = {t_nn_ridge:.4f}, p = {p_nn_ridge:.4f}")
print(f"Ridge vs Baseline: t = {t_ridge_baseline:.4f}, p = {p_ridge_baseline:.4f}")





