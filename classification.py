import pandas as pd
import warnings
import csv
import numpy as np
import ast
import seaborn as sns
from tabulate import tabulate
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

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


X = df[['power', 'toughness', 'price']].values
y = df['rarity'].values

#print(X)
#print(y)

# Two-level cross-validation
k_values = [1, 2, 5, 8, 10]  
K1 = 10 # Outer cross-validation folds
K2 = 10  # Inner cross-validation folds


outer_kfold = KFold(n_splits=K1, shuffle=True, random_state=42)
generalization_errors = []
baseline_errors = []
results_table = []  # To store results for the table

# from sklearn.metrics import classification_report, confusion_matrix
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_train_series = pd.Series(y_train)  # Convert to Pandas Series
# largest_class = y_train_series.mode()[0] 
# baseline_predictions = np.full(y_test.shape, largest_class)

# for x,y in zip(y_test,baseline_predictions):
#     print(x,y)

for i, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(X), 1): #outer fold (loop)
    X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
    y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
    
    # Baseline model: predicting based on the largest class in y_outer_train, thus no parameter training in inner loop required
    y_outer_train_series = pd.Series(y_outer_train)
    largest_class = y_outer_train_series.mode()[0]
    baseline_predictions = np.full(y_outer_test.shape, largest_class)
    accuracy = accuracy_score(y_outer_test, baseline_predictions)
    baseline_loss = 1 - accuracy 
    baseline_errors.append(baseline_loss)

    inner_kfold = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    # To store validation errors for each model with different k
    model_validation_errors = {k: [] for k in k_values}
    
    for inner_train_idx, inner_val_idx in inner_kfold.split(X_outer_train):
        X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
        y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
        
        # Train models for each h value
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            # train model
            model.fit(X_inner_train, y_inner_train)

            # use model to predict y based on X_inner_val
            val_predictions = model.predict(X_inner_val)

            # calc loss
            val_accuracy = accuracy_score(y_inner_val, val_predictions)
            val_loss = 1 - val_accuracy
            model_validation_errors[k].append(val_loss)
    
    # Compute average validation error for each h
    avg_validation_errors = {k: np.mean(model_validation_errors[k]) for k in k_values}
    
    # Select the best model (with lowest validation error)
    best_k = min(avg_validation_errors, key=avg_validation_errors.get)
    print(f"Best k in this outer fold {i}: {best_k}")
    

    # Train the best model on the entire outer training set
    best_model = KNeighborsClassifier(best_k)
    best_model.fit(X_outer_train, y_outer_train)

    # use model to predict y based on X_outer_val
    val_predictions = model.predict(X_outer_test)
    val_accuracy = accuracy_score(y_outer_test, val_predictions)
    val_loss = 1 - val_accuracy
    generalization_errors.append(val_loss)
    
    print(f"Test loss in this outer fold {i}: {val_loss}")
    
#     # Store the results for the table
#     results_table.append([i, best_h, test_loss, baseline_loss])
