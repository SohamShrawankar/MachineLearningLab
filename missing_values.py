import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('house_price_prediction_extended.csv')

# Display null value count before filling
print("Null values before filling:\n", df.isna().sum(), "\n")

# Fill missing values with column mean (vectorized for scalability)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Confirm nulls filled
print("Null values after filling:\n", df.isna().sum(), "\n")

# Define numeric columns for plotting
cols = ['Area (sq ft)', 'Price (Lakhs)']

# Plot boxplots using loop (avoids repetition)
plt.figure(figsize=(10, 4))
for i, col in enumerate(cols, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot - {col}')

plt.tight_layout()
plt.show()
