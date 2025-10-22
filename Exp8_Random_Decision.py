import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load Dataset ---
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
           'BMI','DiabetesPedigreeFunction','Age','Outcome']

df = pd.read_csv(URL, names=COLUMNS)

# --- Features and Target ---
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Models ---
models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# --- Training and Evaluation ---
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Train Accuracy": model.score(X_train, y_train),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# --- Display Results ---
for name, metrics in results.items():
    print(f"\n🔹 {name}")
    print(f"Train Accuracy: {metrics['Train Accuracy']:.3f}")
    print(f"Test Accuracy : {metrics['Test Accuracy']:.3f}")
    print("Confusion Matrix:\n", metrics["Confusion Matrix"])

# --- Feature Importance (Random Forest) ---
rf_clf = models["Random Forest"]
feat_importance = pd.Series(rf_clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
feat_importance.plot(kind='bar', color="skyblue", edgecolor="black")
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
