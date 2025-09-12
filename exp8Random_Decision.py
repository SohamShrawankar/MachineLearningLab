import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']

df = pd.read_csv(url, header=None, names=columns)

df

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_clf = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)
dt_clf.fit(X_train, y_train)

rf_clf=RandomForestClassifier(n_estimators=100,random_state=42)
rf_clf.fit(X_train,y_train)

print("Decision Tree - Training Accuracy",dt_clf.score(X_train,y_train))
print("Decision Tree - Testing Accuracy",dt_clf.score(X_test,y_test))
print("Random Forest - Training Accuracy",rf_clf.score(X_train,y_train))
print("Random Forest - Testing Accuracy",rf_clf.score(X_test,y_test))

importances=rf_clf.feature_importances_
feat_importance=pd.Series(importances,index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
feat_importance.plot(kind='bar',color="skyblue")
plt.title("Feature Importance -  Random Forest")
plt.ylabel("Importance Score")
plt.show()
