import pandas as pd
import warnings
import csv
import numpy as np
import ast
from tabulate import tabulate
from scipy.linalg import svd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 2)
    
def load_data_to_list(file):
    with open(file, 'r') as file:
        reader = csv.reader(file)
        data_list = list(reader)

    return data_list

df = pd.read_csv("Golden_list.csv")

df = df[["name", "legalities", "power", "toughness", "prices", "rarity", "set_id", "artist", "color_identity", "released_at", "keywords", "full_art"]]
df = df.head(1000)
df = df[["power", "toughness", "rarity", "prices"]]
df["power"] = df["power"].replace('*', np.nan)
df["power"] = df["power"].replace('1+*', np.nan)
df = df.dropna()

#print(df["power"].unique())
df[["power"]] = df[["power"]].astype(float)
# print(f"\nPower: {df['power'].apply(type).unique()}")
print(df["power"].unique())

df[["toughness"]] = df[["toughness"]].astype(float)
# print(f"\nThoughness: {df['toughness'].apply(type).unique()}")
# print(df["toughness"].unique())

df["rarity"] = df["rarity"].replace('uncommon', 1)
df["rarity"] = df["rarity"].replace('common', 2)
df["rarity"] = df["rarity"].replace('rare', 3)
df["rarity"] = df["rarity"].replace('mythic', 4)
# print(f"\nRarity: {df['rarity'].apply(type).unique()}")
# print(df["rarity"].unique())

df["price"] = df["prices"].apply(lambda x: ast.literal_eval(x).get('eur'))
df["price"] = pd.to_numeric(df["price"], errors='coerce')
df = df.drop(columns=["prices"])
# print(f"\nPrice: {df['price'].apply(type).unique()}")
# print(df["price"].unique())
print(df.head(3))

means = df.mean()
# print(df.dtypes)
# print(f"\nMeans:\n{means}")
df2 = df[means.index] - means
print(df2.head(3))
# print(f"\nHead 1: \n{df2.head(1)}")

U, S, V = svd(df2, full_matrices=False)
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
# plt.figure()
# plt.plot(range(1, len(rho) + 1), rho, "x-")
# plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
# plt.plot([1, len(rho)], [threshold, threshold], "k--")
# plt.title("Variance explained by principal components")
# plt.xlabel("Principal component")
# plt.ylabel("Variance explained")
# plt.legend(["Individual", "Cumulative", "Threshold"])
# plt.grid()
# plt.show()

Y = df2.to_numpy()

Vt = V.T  # Transpose Vh to get the principal directions

# Project the data onto the principal components
Z = Y @ Vt  # Z contains the data projected onto the principal components

# Indices of the principal components to plot
i, j = 0, 1

# Create the plot
# plt.figure()
# plt.title('PCA: First two principal components')
# plt.scatter(Z[:, i], Z[:, j], alpha=0.7)
# plt.xlabel(f'PC{i+1}')
# plt.ylabel(f'PC{j+1}')
# plt.grid(True)
# plt.show()

# Example: assume Y is your DataFrame
# Convert DataFrame to NumPy array

# Center the data (subtract column means)
Y_centered = Y - np.mean(Y, axis=0)

# PCA using SVD
U, S, Vh = svd(Y_centered, full_matrices=False)
V = Vh.T  # Transpose Vh to get the principal component directions

# Get the names of the features (column names of the original DataFrame)
feature_names = df2.columns

# Plot the coefficients (loadings) for the first four principal components
num_components = 4  # Plot the first four PCs
x = np.arange(len(feature_names))  # x locations for the groups
bar_width = 0.2  # Width of the bars

plt.figure(figsize=(12, 6))

# Create grouped bars for the first 4 PCs
for i in range(num_components):
    plt.bar(x + i * bar_width, V[:, i], bar_width, label=f'PC{i+1}')

# Add labels and formatting
# plt.title('PCA Component Coefficient Plot (First Four Principal Components)')
# plt.xlabel('Features (Attributes)')
# plt.ylabel('Component Coefficients')
# plt.xticks(x + bar_width * 1.5, feature_names, rotation=90)  # Rotate labels for clarity
# plt.legend()
# plt.grid(True)
# plt.tight_layout()  # Adjust layout to avoid label cutoff
# plt.show()

# Assuming your DataFrame is called 'your_dataframe'
# Boxplot of the entire DataFrame
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
df2.boxplot()
plt.title('Boxplot of Original DataFrame')
plt.xlabel('Features (Attributes)')
plt.ylabel('Values')
plt.xticks(rotation=90)  # Rotate labels if feature names are long
plt.grid(True)
plt.tight_layout()  # Adjust layout to avoid label cutoff
plt.show()