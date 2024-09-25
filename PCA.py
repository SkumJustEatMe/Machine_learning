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
pd.set_option('display.max_columns', 3)
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
df = df.dropna()

df[["power"]] = df[["power"]].astype(float)
# print(f"\nPower: {df['power'].apply(type).unique()}")
# print(df["power"].unique())

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

means = df.mean()
# print(df.dtypes)
# print(f"\nMeans:\n{means}")
df2 = df[means.index] - means
# print(f"\nHead 1: \n{df2.head(1)}")

U, S, V = svd(df2, full_matrices=False)
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()