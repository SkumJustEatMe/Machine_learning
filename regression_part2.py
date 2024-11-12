import pandas as pd
import warnings
import numpy as np
import ast
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.optim import Adam

from scipy import stats
from scipy.stats import ttest_rel
from scipy.stats import t

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

df2 = (df[means.index] - means) / stds  # normalized dataset


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


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, epochs):
    criterion = nn.MSELoss()
    learning_rate = 0.01  
    
    ## train model on training data
    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        
        ## update parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad   
    
    ## evaluate trained model on test data 
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val).item()
    
    return val_loss

def compute_confidence_interval(zi, se, n, alpha=0.05):
    critical_t = t.ppf(q=1 - alpha/2, df = n-1) #inverse of cdf, percent point function
    margin_of_error = critical_t * se
    CI_u = zi + margin_of_error
    CI_l = zi - margin_of_error
    return CI_u, CI_l

# TWO LEVEL CROSS VAL --------------------------------------------------------------------------------------------

##PARAMS
h_values = [1, 2, 5, 8, 10]  
lambdas = [10**(-4), 10**(-2), 1, 10, 100]  #
K1 = 10  # outer
K2 = 10  # inner

epochs = 500

outer_kfold = KFold(n_splits=K1, shuffle=True, random_state=42)
results_table = []

for i, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(X), 1):  # outer fold (loop)
    X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
    y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
    
    ## baseline model: predicting the mean of y_outer_train, no parameter training
    baseline_mean = y_outer_train.mean()
    baseline_predictions = np.full((len(y_outer_test), 1), baseline_mean)  # Predicting mean for all test samples
    baseline_loss = np.mean((baseline_predictions.flatten() - y_outer_test) ** 2)

    inner_kfold = KFold(n_splits=K2, shuffle=True, random_state=42)
    
    model_validation_errors = {h: [] for h in h_values}
    ridge_validation_errors = {lamb: [] for lamb in lambdas}

    for inner_train_idx, inner_val_idx in inner_kfold.split(X_outer_train):
        X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
        y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
        
        ## train for each h 
        for h in h_values:
            model = PricePredictionANN(h)
            val_loss = train_and_evaluate_model(model, torch.tensor(X_inner_train, dtype=torch.float32), 
                                                 torch.tensor(y_inner_train, dtype=torch.float32).view(-1, 1), 
                                                 torch.tensor(X_inner_val, dtype=torch.float32), 
                                                 torch.tensor(y_inner_val, dtype=torch.float32).view(-1, 1), 
                                                 epochs)
            model_validation_errors[h].append(val_loss)

        ## train for each lamb
        for lamb in lambdas:
            ridge_model = Ridge(alpha=lamb)
            ridge_model.fit(X_inner_train, y_inner_train)
            val_loss_ridge = np.mean((ridge_model.predict(X_inner_val) - y_inner_val) ** 2)
            ridge_validation_errors[lamb].append(val_loss_ridge)

    ## calc generalization error for eaach h and lamb
    avg_validation_errors = {h: np.mean(model_validation_errors[h]) for h in h_values}
    avg_ridge_errors = {lamb: np.mean(ridge_validation_errors[lamb]) for lamb in lambdas}

    ## select best model
    best_h = min(avg_validation_errors, key=avg_validation_errors.get)
    
    ## train and eval best models
    best_model = PricePredictionANN(best_h)
    test_loss_ann = train_and_evaluate_model(best_model, torch.tensor(X_outer_train, dtype=torch.float32), 
                              torch.tensor(y_outer_train, dtype=torch.float32).view(-1, 1), 
                              torch.tensor(X_outer_train, dtype=torch.float32), 
                              torch.tensor(y_outer_train, dtype=torch.float32).view(-1, 1), 
                              epochs)
    

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

    results_table.append([i, best_h, test_loss_ann, baseline_loss, best_lambda, best_ridge_loss])



## convert results to df ------------------------------------------------
results_df = pd.DataFrame(results_table, columns=["Outer Fold", "Best h (ANN)", "E_test (Neural Net)", "E_test (Baseline)", "Best λ (Ridge)", "E_test (Ridge)"])
print("\nResults Table:\n")
print(results_df)

results_df.to_csv('results_table.csv', index=False)

## STATS ---------------------------------------------------------------


results_df = pd.read_csv("results_table.csv")

e_test_nn = results_df['E_test (Neural Net)']
e_test_baseline = results_df['E_test (Baseline)']
e_test_ridge = results_df['E_test (Ridge)']

t_nn_baseline, p_nn_baseline = ttest_rel(e_test_nn, e_test_baseline)
t_nn_ridge, p_nn_ridge = ttest_rel(e_test_nn, e_test_ridge)
t_ridge_baseline, p_ridge_baseline = ttest_rel(e_test_ridge, e_test_baseline)

differences_nn_baseline = e_test_nn - e_test_baseline
differences_nn_ridge = e_test_nn - e_test_ridge
differences_ridge_baseline = e_test_ridge - e_test_baseline

mean_diff_nn_baseline = np.mean(differences_nn_baseline)
mean_diff_nn_ridge = np.mean(differences_nn_ridge)
mean_diff_ridge_baseline = np.mean(differences_ridge_baseline)

n = len(differences_nn_baseline)
std_diff_nn_baseline = np.sqrt(np.sum((differences_nn_baseline - mean_diff_nn_baseline)**2) / (n * (n - 1)))
std_diff_nn_ridge = np.sqrt(np.sum((differences_nn_ridge - mean_diff_nn_ridge)**2) / (n * (n - 1)))
std_diff_ridge_baseline = np.sqrt(np.sum((differences_ridge_baseline - mean_diff_ridge_baseline)**2) / (n * (n - 1)))

alpha = 0.05
confidence_level = 1 - alpha
t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

## CI 
ci_lower_nn_baseline = mean_diff_nn_baseline - t_critical * std_diff_nn_baseline
ci_upper_nn_baseline = mean_diff_nn_baseline + t_critical * std_diff_nn_baseline

ci_lower_nn_ridge = mean_diff_nn_ridge - t_critical * std_diff_nn_ridge
ci_upper_nn_ridge = mean_diff_nn_ridge + t_critical * std_diff_nn_ridge

ci_lower_ridge_baseline = mean_diff_ridge_baseline - t_critical * std_diff_ridge_baseline
ci_upper_ridge_baseline = mean_diff_ridge_baseline + t_critical * std_diff_ridge_baseline

## export----

# Data for the table
results_df = {
    "Comparison": [
        "Neural Net vs Baseline",
        "Neural Net vs Ridge",
        "Ridge vs Baseline"
    ],
    "p-value": [
        p_nn_baseline,
        p_nn_ridge,
        p_ridge_baseline
    ],
    "Mean Difference": [
        mean_diff_nn_baseline,
        mean_diff_nn_ridge,
        mean_diff_ridge_baseline
    ],
    "Confidence Interval Lower": [
        ci_lower_nn_baseline,
        ci_lower_nn_ridge,
        ci_lower_ridge_baseline
    ],
    "Confidence Interval Upper": [
        ci_upper_nn_baseline,
        ci_upper_nn_ridge,
        ci_upper_ridge_baseline
    ]
}


results_table = pd.DataFrame(results_df)

results_table.to_csv("statistical_analysis_results.csv", index=False)
