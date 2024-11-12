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
from scipy.stats import binom
from scipy.stats import beta

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

## LOAD DATA ----------------
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

k_values = [1, 2, 5, 8, 10]  
C_values = [0.001, 0.01, 0.1, 1, 10] 
K1 = 10 ## outer
K2 = 10  ## inner


outer_kfold = KFold(n_splits=K1, shuffle=True, random_state=42)
generalization_errors = []
baseline_errors = []
results_table = []  

stats_calculations_kn_lr_n12 = []
stats_calculations_kn_lr_n21 = []
stats_calculations_kn_base_n12 = []
stats_calculations_kn_base_n21 = []
stats_calculations_lr_base_n12 = []
stats_calculations_lr_base_n21 = []


def compute_confidence_interval(n12,n21,n, alpha=0.05):
    E_theta = (n12 - n21) / n
    
    Q_numerator =  n**2 * (n + 1) * (E_theta + 1) * (1 - E_theta)
    Q_denominator =  n * (n12 + n21) - (n12 - n21)**2
    Q = Q_numerator/Q_denominator

    f = (E_theta + 1) / 2 * (Q - 1)
    g = (1 - E_theta) / 2 * (Q - 1)

    theta_L = 2 * beta.ppf(alpha / 2, f, g) - 1
    theta_U = 2 * beta.ppf(1 - alpha / 2, f, g) - 1 

    return theta_L, theta_U, E_theta

def compute_p_values(n12,n21):
    m = min(n12,n21)
    theta = 1/2
    N = n12 + n21
    p_value = 2 * binom.cdf(k=m,n=N,p=theta)
    return p_value


for i, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(X), 1): #outer fold (loop)
    X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
    y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
    
    # base calc 
    y_outer_train_series = pd.Series(y_outer_train)
    largest_class = y_outer_train_series.mode()[0]
    baseline_predictions = np.full(y_outer_test.shape, largest_class)
    accuracy = accuracy_score(y_outer_test, baseline_predictions)
    baseline_loss = 1 - accuracy 
    baseline_errors.append(baseline_loss)

    inner_kfold = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    

    # inner loop init 
    knn_validation_errors = {k: [] for k in k_values}
    lr_validation_errors = {c: [] for c in C_values}
    
    for inner_train_idx, inner_val_idx in inner_kfold.split(X_outer_train):
        X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
        y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
        
        # TRAIN MODELS 
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
    
    # calc generalization error 
    #avg_validation_errors = {k: np.mean(knn_validation_errors[k]) for k in k_values}
    avg_knn_errors = {k: np.mean(knn_validation_errors[k]) for k in k_values}
    avg_lr_errors = {c: np.mean(lr_validation_errors[c]) for c in C_values}
    
    # select best model
    best_k = min(avg_knn_errors, key=avg_knn_errors.get)
    best_c = min(avg_lr_errors, key=avg_lr_errors.get)
    
    # train and test best models
    best_knn_model = KNeighborsClassifier(best_k)
    best_knn_model.fit(X_outer_train, y_outer_train)

    best_lr_model = LogisticRegression(C=best_c, max_iter=1000)
    best_lr_model.fit(X_outer_train, y_outer_train)
    knn_test_predictions = best_knn_model.predict(X_outer_test)
    knn_test_accuracy = accuracy_score(y_outer_test, knn_test_predictions) # number of correctly classified out of n predictions
    knn_test_loss = 1 - knn_test_accuracy # number of incorrectly classified out of n predictions

    lr_test_predictions = best_lr_model.predict(X_outer_test)
    lr_test_accuracy = accuracy_score(y_outer_test, lr_test_predictions)
    lr_test_loss = 1 - lr_test_accuracy

    ## Store the results for the table
    results_table.append([i, best_k, best_c, knn_test_loss,lr_test_loss, baseline_loss])


    ## Data collection for statistics 
    lr_model_correct = (lr_test_predictions == y_outer_test)
    kn_model_correct = (knn_test_predictions == y_outer_test)
    base_model_correct = (baseline_predictions == y_outer_test)

    n_predictions = len(y_outer_test)

    # KNN correct, Logistic Regression wrong
    kn_correct_lr_wrong = np.sum(kn_model_correct & ~lr_model_correct) #n12

    # KNN wrong, Logistic Regression correct
    kn_wrong_lr_correct = np.sum(~kn_model_correct & lr_model_correct) #n21

    # KNN correct, Baseline wrong
    kn_correct_base_wrong = np.sum(kn_model_correct & ~base_model_correct) 
    
    # KNN wrong, Baseline correct
    kn_wrong_base_correct = np.sum(~kn_model_correct & base_model_correct)

    # Logistic Regression correct, Baseline wrong
    lr_correct_base_wrong = np.sum(lr_model_correct & ~base_model_correct)

    # Logistic Regression wrong, Baseline correct
    lr_wrong_base_correct = np.sum(~lr_model_correct & base_model_correct)
    
    # calc CI and estimated diff 
    #kn_lr_CI_lower, kn_lr_CI_upper, estimated_difference_kn_lr = compute_confidence_interval(n12 = kn_correct_lr_wrong, n21 = kn_wrong_lr_correct, n = n_predictions)
    #kn_base_CI_lower, kn_base_CI_upper, estimated_diiference_kn_base = compute_confidence_interval(n12=kn_correct_base_wrong, n21 = kn_wrong_base_correct, n=n_predictions)
    #lr_base_CI_lower, lr_base_CI_upper, estimated_difference_lr_base = compute_confidence_interval(n12=lr_correct_base_wrong, n21 = lr_wrong_base_correct, n = n_predictions)

    # calc pvalues 
    #kn_lr_pvalue = compute_p_values(n12 = kn_correct_lr_wrong, n21 = kn_wrong_lr_correct)
    #kn_base_pvalue = compute_p_values(n12=kn_correct_base_wrong, n21 = kn_wrong_base_correct)
    #lr_base_pvalue = compute_p_values(n12=lr_correct_base_wrong, n21 = lr_wrong_base_correct)

    stats_calculations_kn_lr_n12.append(kn_correct_lr_wrong)
    stats_calculations_kn_lr_n21.append(kn_wrong_lr_correct)

    stats_calculations_kn_base_n12.append(kn_correct_base_wrong)
    stats_calculations_kn_base_n12.append(kn_wrong_base_correct)

    stats_calculations_lr_base_n12.append(lr_correct_base_wrong)
    stats_calculations_lr_base_n21.append(lr_wrong_base_correct)



    #stats_calculations_kn_lr.append([i, kn_lr_CI_lower,kn_lr_CI_upper,estimated_difference_kn_lr,kn_lr_pvalue])
    #stats_calculations_kn_base.append([i,kn_base_CI_lower, kn_base_CI_upper, estimated_diiference_kn_base,kn_base_pvalue])
    #stats_calculations_lr_base.append([i,lr_base_CI_lower, lr_base_CI_upper, estimated_difference_lr_base,lr_base_pvalue])

results_df = pd.DataFrame(results_table, columns=["Outer Fold", "Best k", "Best lambda", "E_test (KNN)", "E_test (Logistic regression)", "E_test (baseline_loss)"])
print("\nResults Table:\n")
print(results_df)

results_df.to_csv('results_table_classification.csv', index=False)

# DO STATS CALC ----------------------
kn_lr_pvalue = compute_p_values(n12 = np.sum(stats_calculations_kn_lr_n12), n21 = np.sum(stats_calculations_kn_lr_n21))
kn_lr_CI_lower, kn_lr_CI_upper, kn_lr_estimated_diff = compute_confidence_interval(n12 = np.sum(stats_calculations_kn_lr_n12), n21 = np.sum(stats_calculations_kn_lr_n21), n = n_predictions*10)

#print("\nKNN vs Base Stats:\n")
kn_base_pvalue = compute_p_values(n12 = np.sum(stats_calculations_kn_base_n12), n21 = np.sum(stats_calculations_kn_base_n21))
kn_base_CI_lower, kn_base_CI_upper, kn_base_estimated_diff = compute_confidence_interval(n12 = np.sum(stats_calculations_kn_base_n12), n21 = np.sum(stats_calculations_kn_base_n21), n = n_predictions*10)

lr_base_pvalue = compute_p_values(n12 = np.sum(stats_calculations_lr_base_n12), n21 = np.sum(stats_calculations_lr_base_n21))
lr_base_CI_lower, lr_base_CI_upper, lr_base_estimated_diff = compute_confidence_interval(n12 = np.sum(stats_calculations_lr_base_n12), n21 = np.sum(stats_calculations_lr_base_n21), n = n_predictions*10)


stats_total = [["kn_lr",kn_lr_pvalue,kn_lr_estimated_diff,kn_lr_CI_lower,kn_lr_CI_upper],["kn_base",kn_base_pvalue,kn_base_estimated_diff,kn_base_CI_lower,kn_base_CI_upper],["lr_base",lr_base_pvalue,lr_base_estimated_diff,lr_base_CI_lower,lr_base_CI_upper]]
stats_total_df = pd.DataFrame(stats_total,columns=["Comparisons", "P-value","Estimated diff","CI lower", "CI upper"])
stats_total_df.to_csv('stats_results_classification.csv',index=False)

# Logistically regression model training

class_names = ['Uncommon', 'Common', 'Rare', 'Mythic']

model = LogisticRegression(C=10, max_iter=1000)
model.fit(X_outer_train, y_outer_train)

feature_names = ['power', 'toughness', 'price']
coefficients = model.coef_

coef_df = pd.DataFrame(coefficients, columns=feature_names)
coef_df['Class'] = [class_names[i] for i in range(len(coefficients))]

print("Coefficients per class:")
print(coef_df)



