import numpy as np 
import pandas as pd

np.random.seed(42)

def _late_night_weights():
    """Higher probability for late-night/early-morning hours (fraud pattern)."""
    weights = np.ones(24)
    weights[0:5]   = 3.0   # midnight - 5am: heavily weighted
    weights[22:24] = 2.0   # 10pm - midnight: moderately weighted
    return weights / weights.sum()


n_legit = 9800
n_fraud = 200

legit = pd.DataFrame({
    "transaction_amount":        np.random.gamma(2, 50, n_legit),
    "transaction_hour":          np.random.randint(0, 24, n_legit),
    "account_age_days":          np.random.exponential(500, n_legit),
    "num_transactions_last_24h": np.random.poisson(2, n_legit),
    "distance_from_home_km":     np.random.exponential(15, n_legit),
})
legit["is_fraud"] = 0


fraud = pd.DataFrame({
    "transaction_amount":        np.random.gamma(2, 55, n_fraud),
    "transaction_hour":          np.random.choice(
                                      range(24), n_fraud,
                                      p=_late_night_weights()
                                  ),
    "account_age_days":          np.random.exponential(60, n_fraud),
    "num_transactions_last_24h": np.random.poisson(8, n_fraud),
    "distance_from_home_km":     np.random.exponential(45, n_fraud),
})
fraud["is_fraud"] = 1


df = pd.concat([legit, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total transactions: {len(df)}")
print(f"\nClass distribution:")
print(df["is_fraud"].value_counts())
print(f"\nFraud percentage: {df['is_fraud'].mean()*100:.2f}%")


df["transaction_amount"]    = df["transaction_amount"].round(2)
df["account_age_days"]      = df["account_age_days"].round(0).astype(int)
df["distance_from_home_km"] = df["distance_from_home_km"].round(2)

df.to_csv("data/raw_transactions.csv", index=False)
print("\n✓ Saved → data/raw_transactions.csv")
print(df.head())