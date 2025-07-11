pip install numpy pandas matplotlib seaborn scikit-learn

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.head())



print(df.head())
print(df.info())
print(df.describe())
print(df['species'].value_counts())

sns.pairplot(df, hue='species')
plt.suptitle("Pair plot of iris features", y=1.02)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, orient ="h")
plt.title("Box plot of all features")
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True , cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


