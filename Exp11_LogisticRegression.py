import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a simple dataset for demonstration
data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [0.5, 1.2, 2.1, 3.0, 4.5, 5.1, 6.0, 7.2, 8.1, 9.0],
    'Target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # Binary target variable
}
df = pd.DataFrame(data)

X = df[['Feature1', 'Feature2']] # Features
y = df['Target'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression(solver='liblinear', random_state=42) # 'liblinear' is a good choice for small datasets
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)


y_prob = model.predict_proba(X_test)[:, 1] # Probability of the positive class (1)
print("\nPredicted Probabilities for Class 1:")
print(y_prob)
