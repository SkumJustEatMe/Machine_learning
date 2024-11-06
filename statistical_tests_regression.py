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
from scipy import stats

## statistical analysis 

# Load df
results_df = pd.read_csv("results_table.csv")

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


differences_nn_baseline = e_test_nn - e_test_baseline
differences_nn_ridge = e_test_nn - e_test_ridge
differences_ridge_baseline = e_test_ridge - e_test_baseline

# Mean of differences
mean_diff_nn_baseline = np.mean(differences_nn_baseline)
mean_diff_nn_ridge = np.mean(differences_nn_ridge)
mean_diff_ridge_baseline = np.mean(differences_ridge_baseline)

# Standard deviation of differences
n = len(differences_nn_baseline)
std_diff_nn_baseline = np.sqrt(np.sum((differences_nn_baseline - mean_diff_nn_baseline)**2) / (n * (n - 1)))
std_diff_nn_ridge = np.sqrt(np.sum((differences_nn_ridge - mean_diff_nn_ridge)**2) / (n * (n - 1)))
std_diff_ridge_baseline = np.sqrt(np.sum((differences_ridge_baseline - mean_diff_ridge_baseline)**2) / (n * (n - 1)))

# Confidence level
alpha = 0.05
confidence_level = 1 - alpha

# Critical t-value
t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

# Confidence interval
ci_lower_nn_baseline = mean_diff_nn_baseline - t_critical * std_diff_nn_baseline
ci_upper_nn_baseline = mean_diff_nn_baseline + t_critical * std_diff_nn_baseline

ci_lower_nn_ridge = mean_diff_nn_ridge - t_critical * std_diff_nn_ridge
ci_upper_nn_ridge = mean_diff_nn_ridge + t_critical * std_diff_nn_ridge

ci_lower_ridge_baseline = mean_diff_ridge_baseline - t_critical * std_diff_ridge_baseline
ci_upper_ridge_baseline = mean_diff_ridge_baseline + t_critical * std_diff_ridge_baseline

print("Mean difference (Neural Net vs Baseline):", mean_diff_nn_baseline)
print(f"{int(confidence_level * 100)}% Confidence Interval: [{ci_lower_nn_baseline}, {ci_upper_nn_baseline}]") # where we expect the true difference in E_test to lie

print("Mean difference (Neural Net vs Ridge):", mean_diff_nn_ridge)
print(f"{int(confidence_level * 100)}% Confidence Interval: [{ci_lower_nn_ridge}, {ci_upper_nn_ridge}]")

print("Mean difference (Neural Net vs Ridge):", mean_diff_ridge_baseline)
print(f"{int(confidence_level * 100)}% Confidence Interval: [{ci_lower_ridge_baseline}, {ci_upper_ridge_baseline}]")

## export

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

# Create a DataFrame
results_table = pd.DataFrame(results_df)

# Save to CSV
results_table.to_csv("statistical_analysis_results.csv", index=False)
