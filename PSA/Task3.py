# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
houseData = pd.read_csv("data/housing_training.csv")
A = houseData['a']

# %%
sns.histplot(A)
