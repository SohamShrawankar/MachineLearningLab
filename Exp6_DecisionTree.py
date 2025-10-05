import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


warnings.filterwarnings("ignore")

def main():

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]


    df = pd.read_csv(url, names=cols)


    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42))
    ])


    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)


    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"\nAverage Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


    clf = pipeline.named_steps['clf']
    plt.figure(figsize=(16, 8))
    plot_tree(clf, feature_names=X.columns, class_names=["Non-Diabetic", "Diabetic"],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree (Entropy, max_depth=4)")
    plt.show()

if __name__ == "__main__":
    main()
