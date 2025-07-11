import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("house_price_prediction.csv")

df_sal.describe()

print(df.head())

X = df[['Area (sq ft)']] 
y = df['Price (Lakhs)']


model = LinearRegression()


model.fit(X, y)

y_pred = model.predict(X)


r2 = r2_score(y, y_pred)
print(f"R² Score: {r2:.4f}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Simple Linear Regression - House Price Prediction")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakhs)")
plt.legend()
plt.grid(True)
plt.show()


