import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("house_price_prediction_PCA.csv")

df.describe()

# Fill missing values
df[['Area (sq ft)', 'Price (Lakhs)']] = df[['Area (sq ft)', 'Price (Lakhs)']].fillna(df[['Area (sq ft)', 'Price (Lakhs)']].mean())

# Group by Label and compute average price
avg_price_per_label = df.groupby('Label')['Price (Lakhs)'].mean().reset_index()

# Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=avg_price_per_label, x='Label', y='Price (Lakhs)', palette='Set2')
plt.title('Average Price by Label')
plt.ylabel('Average Price (Lakhs)')
plt.xlabel('Label')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

 print(df.head())


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("house_price_prediction_PCA.csv")

# Step 1: Select only numeric columns (e.g., 'Area', 'Price', etc.)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Step 2: Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(numeric_df)

# Step 3: Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)

# Step 4: Apply PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Step 5: Convert PCA result to DataFrame
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Step 6: Re-attach the original label columns (if present)
for col in ['Label', 'Label_Numeric']:
    if col in df.columns:
        pca_df[col] = df[col]

# Step 7: Output
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print(pca_df.head())





eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)



LABEL ENCODING 

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Label'])


print(df[['Label', 'Label_Encoded']].head())
