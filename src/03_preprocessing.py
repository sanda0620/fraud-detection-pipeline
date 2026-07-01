import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("data/raw_transactions.csv")
print(f"Loaded: {df.shape}")

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print(f"\nFeatures (X): {X.shape}")
print(f"Target   (y): {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")
print(f"Fraud cases in y: {y.sum()} ({y.mean()*100:.2f}%)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n── Train/Test Split ──")
print(f"Training set:   {X_train.shape[0]} rows")
print(f"Test set:       {X_test.shape[0]} rows")

print("\nFraud distribution after split:")
print(f"  Train fraud: {y_train.sum()} / {len(y_train)} ({y_train.mean()*100:.2f}%)")
print(f"  Test fraud:  {y_test.sum()} / {len(y_test)} ({y_test.mean()*100:.2f}%)")


print("\n── Feature Scaling ──")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns)

print("Scaler fitted on training data only.")
print(f"\nBefore scaling (train means):")
print(X_train.mean().round(2))
print(f"\nAfter scaling (train means):")
print(X_train_scaled.mean().round(4))
print(f"\nAfter scaling (train std devs):")
print(X_train_scaled.std().round(4))


print("\n── Saving processed data ──")

X_train_scaled.to_csv("data/X_train.csv", index=False)
X_test_scaled.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv",         index=False)
y_test.to_csv("data/y_test.csv",           index=False)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("  data/X_train.csv  ✓")
print("  data/X_test.csv   ✓")
print("  data/y_train.csv  ✓")
print("  data/y_test.csv   ✓")
print("  models/scaler.pkl ✓")