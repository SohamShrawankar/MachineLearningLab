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

while True:
    try:
        user_input = input("Enter house area in sq ft (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        area_value = float(user_input)
        predicted_price = model.predict([[area_value]])
        print(f"Predicted Price for {area_value} sq ft = ₹{predicted_price[0]:.2f} Lakhs\n")
    except ValueError:
        print("Invalid input. Please enter a valid number.\n")
