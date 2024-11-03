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
C_values = [0.001, 0.01, 0.1, 1, 10] 
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
    knn_validation_errors = {k: [] for k in k_values}
    lr_validation_errors = {c: [] for c in C_values}
    
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
            knn_validation_errors[k].append(val_loss)

        for c in C_values:
            model = LogisticRegression(C=c, max_iter=1000)
            model.fit(X_inner_train, y_inner_train)
            val_predictions = model.predict(X_inner_val)
            val_accuracy = accuracy_score(y_inner_val, val_predictions)
            val_loss = 1 - val_accuracy
            lr_validation_errors[c].append(val_loss)
    
    # Compute average validation error for each h
    #avg_validation_errors = {k: np.mean(knn_validation_errors[k]) for k in k_values}
    avg_knn_errors = {k: np.mean(knn_validation_errors[k]) for k in k_values}
    avg_lr_errors = {c: np.mean(lr_validation_errors[c]) for c in C_values}
    
    # Select the best model (with lowest validation error)
    best_k = min(avg_knn_errors, key=avg_knn_errors.get)
    best_c = min(avg_lr_errors, key=avg_lr_errors.get)

    print(f"Best k in this outer fold {i}: {best_k}")
    print(f"Best C in this outer fold {i}: {best_c}")

    # # Train the best model on the entire outer training set
    # best_model = KNeighborsClassifier(best_k)
    # best_model.fit(X_outer_train, y_outer_train)
    
    # Train the best KNN model on the entire outer training set
    best_knn_model = KNeighborsClassifier(best_k)
    best_knn_model.fit(X_outer_train, y_outer_train)

    # Train the best Logistic Regression model on the entire outer training set
    best_lr_model = LogisticRegression(C=best_c, max_iter=1000)
    best_lr_model.fit(X_outer_train, y_outer_train)
    knn_test_predictions = best_knn_model.predict(X_outer_test)
    knn_test_accuracy = accuracy_score(y_outer_test, knn_test_predictions) # number of correctly classified out of n predictions
    knn_test_loss = 1 - knn_test_accuracy # number of incorrectly classified out of n predictions
    print(f"KNN Test loss in this outer fold {i}: {knn_test_loss}")

    # Evaluate the best Logistic Regression model on the outer test set
    lr_test_predictions = best_lr_model.predict(X_outer_test)
    lr_test_accuracy = accuracy_score(y_outer_test, lr_test_predictions)
    lr_test_loss = 1 - lr_test_accuracy
    print(f"Logistic Regression Test loss in this outer fold {i}: {lr_test_loss}")
    
    print(f"Test loss in this outer fold {i}: {val_loss}")
    
    # Store the results for the table
    results_table.append([i, best_k, best_c, knn_test_loss,lr_test_loss, baseline_loss])

# Display results in a table
results_df = pd.DataFrame(results_table, columns=["Outer Fold", "Best k", "Best lambda", "E_test (KNN)", "E_test (Logistic regression)", "E_test (baseline_loss)"])
print("\nResults Table:\n")
print(results_df)

# Optionally save the results to a CSV file
results_df.to_csv('results_table_classification.csv', index=False)
