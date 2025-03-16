# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
houseData = pd.read_csv("data/housing_training.csv")
houseData.head()

# %%
k = houseData['k']
m = houseData['m']
n = houseData['n']

# %%
sns.violinplot(k,native_scale=True)

# %%
sns.violinplot(m,native_scale=True)

# %%
sns.violinplot(n,native_scale=True)
