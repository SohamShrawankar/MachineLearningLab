import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([[1,1], [2,1], [3,2], [4,3], [5,3],])
y = np.array([20,40,50,65,80])

model = LinearRegression()
model.fit(X, y)
LinearRegression()

y_pred = model.predict(X)

residuals = y - y_pred

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rsme = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("Coefficients (b1=Hours, b2=practice_Tests):", model.coef_)
print("intercept (B0):", model.intercept_)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared error (MSE):", mse)
print("Root Mean Squared error (RSME):", rsme)
print("R2 score (R2 score):", r2)

plt.scatter(y_pred, residuals, color='blue', label="Residuals")
plt.axhline(y=0, color='red', linestyle="--")
plt.xlabel("Predicted Scores")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Analysis - Multiple Linear Regression (Students)")
plt.legend()
plt.show()
