import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_transactions.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")



print("\n── Class Distribution ──")
class_counts = df["is_fraud"].value_counts()
class_pct    = df["is_fraud"].value_counts(normalize=True).mul(100).round(2)

dist_df = pd.DataFrame({
    "count": class_counts,
    "percent": class_pct
})
dist_df.index = ["Legitimate (0)", "Fraud (1)"]
print(dist_df)

imbalance_ratio = class_counts[0] / class_counts[1]
print(f"\nImbalance ratio → 1 fraud case for every {imbalance_ratio:.1f} legit cases")



print("\n── The Accuracy Trap (proof) ──")
majority_class_predictions = np.zeros(len(df))  # predict "not fraud" always
naive_accuracy = (majority_class_predictions == df["is_fraud"]).mean()

print(f"A model that ALWAYS predicts 'not fraud':")
print(f"  Accuracy: {naive_accuracy*100:.2f}%")
print(f"  Fraud cases caught: 0 out of {df['is_fraud'].sum()}")
print(f"  This model is USELESS despite the high accuracy score.")


print("\n── Feature Comparison: Legit vs Fraud ──")
numeric_cols = ["transaction_amount", "transaction_hour", "account_age_days",
                 "num_transactions_last_24h", "distance_from_home_km"]

comparison = df.groupby("is_fraud")[numeric_cols].mean().round(2)
comparison.index = ["Legitimate", "Fraud"]
print(comparison)


print("\n── Feature Spread: Legit vs Fraud (std dev) ──")
spread = df.groupby("is_fraud")[numeric_cols].std().round(2)
spread.index = ["Legitimate", "Fraud"]
print(spread)


print("\n── Correlation with is_fraud ──")
correlations = df[numeric_cols + ["is_fraud"]].corr()["is_fraud"].drop("is_fraud")
correlations = correlations.sort_values(key=abs, ascending=False)
print(correlations.round(3))


print("\n── EDA Summary ──")
print(f"Total transactions: {len(df)}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
print(f"Imbalance ratio: 1:{imbalance_ratio:.0f}")
print(f"\nStrongest fraud predictor: {correlations.index[0]} (corr={correlations.iloc[0]:.3f})")
print(f"Weakest fraud predictor:   {correlations.index[-1]} (corr={correlations.iloc[-1]:.3f})")