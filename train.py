import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)

# 1. Load Dataset 
data = load_digits()
X = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
y = pd.Series(data.target)

print(f"Total Features: {X.shape[1]} (Required >= 12)")
print(f"Total Instances: {X.shape[0]} (Required >= 500)")

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Save Test Dataset CSV for GitHub & App testing
test_data = X_test.copy()
test_data['target'] = y_test.values
test_data.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully!")

# 5. Create Model Directory
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")

# 6. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# 7. Train, Evaluate, and Print Metrics
print("\n--- Training and Evaluating Models ---")
for name, model in models.items():
    if name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    # Calculate metrics (using 'weighted' average for multi-class support)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)
    
    print(f"\nModel: {name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    
    # Save trained model artifact
    filename = f"model/{name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, filename)
print("\nAll models trained, evaluated, and saved  in model/")