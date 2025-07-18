import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('house_price_prediction_extended.csv')

# Step 1: Check for null values
print("Null values before filling:\n", df.isnull().sum())

# Step 2: Fill null values with column mean
df['Area (sq ft)'] = df['Area (sq ft)'].fillna(df['Area (sq ft)'].mean())
df['Price (Lakhs)'] = df['Price (Lakhs)'].fillna(df['Price (Lakhs)'].mean())

# Confirm filling
print("\nNull values after filling:\n", df.isnull().sum())

# Step 3: Plot box plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['Area (sq ft)'])
plt.title('Box Plot - Area (sq ft)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Price (Lakhs)'])
plt.title('Box Plot - Price (Lakhs)')

plt.tight_layout()
plt.show()
