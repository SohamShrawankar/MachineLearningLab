import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

cols = ['Pregnancies', 'Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' , 'DiabetesPedigreeFunction' ,'Age' , 'Outcome']

cols

df = pd.read_csv(url, names=cols)

X= df.drop('Outcome', axis=1)
y = df['Outcome']

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=42)

clf = DecisionTreeClassifier(criterion='entropy' , max_depth=3 , random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:" , accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test , y_pred))

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["Non-Diabetic", "Diabetic"] , filled=True)
plt.show()
