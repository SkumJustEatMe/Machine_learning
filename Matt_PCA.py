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
df = df.head(8)
print(df)

print(df.mean())

means = df.mean()
df2 = df[means.index] - means
print(df2)

#SSD = Sum of squared distances.
print(f"Sum of Squared Distances for Adjusted Power: {(df2['power']**2).sum()}")
print(f"Sum of Squared Distances for Adjusted Price: {(df2['price']**2).sum()}")

# Singular value decomposition

# Create a 3D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df2["power"], df2["price"], color='r', s=100)

# Labels and title
plt.xlabel('Power')
plt.ylabel('Price')
plt.title('2D Scatter Plot of Price vs. Power')

# Show the plot
plt.grid(True)
plt.show()