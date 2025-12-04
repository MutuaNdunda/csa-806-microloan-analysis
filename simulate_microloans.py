"""
simulate_microloans.py

Generates a synthetic microloan transaction dataset with:
- N_ROWS rows
- 500 numeric features named feat_000 .. feat_499
- binary target column 'default' correlated to a small set of causal features

Outputs:
- CSV file (gzip-compressed) named microloans_{N_ROWS}.csv.gz

Usage:
    python simulate_microloans.py --n_rows 200000 --out microloans_200k.csv.gz
"""

import numpy as np
import pandas as pd
import argparse
import os
from scipy.special import expit

def generate_dataset(n_rows=200_000, n_features=500, seed=42, out="microloans.csv.gz"):
    rng = np.random.default_rng(seed)

    # Base noise for all features
    X = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_features)).astype(np.float32)

    # create a few "causal" features that influence default
    n_causal = 12
    causal_idx = np.arange(n_causal)  # first 12 features will be causal
    weights = rng.normal(0.8, 0.6, size=n_causal)  # effect sizes

    # linear score from causal features
    linear_score = X[:, causal_idx] @ weights
    # add some non-linear and seasonal effects
    month = rng.integers(1, 13, size=n_rows)
    seasonal_effect = np.sin(month / 12.0 * 2 * np.pi) * 0.5
    score = linear_score + seasonal_effect + rng.normal(0, 1.2, size=n_rows)

    # convert score to probability of default using logistic function
    prob_default = expit((score - np.mean(score)) / (np.std(score) + 1e-9)) * 0.5  # keep default rate moderate
    default = rng.random(n_rows) < prob_default
    default = default.astype(int)

    # add a few engineered features (counts, ratios)
    X[:, 100] += (month - 6) * 0.1  # seasonal shift on one feature
    X[:, 101] = X[:, :3].sum(axis=1)  # small engineered feature

    # Assemble dataframe in a memory-efficient way
    col_names = [f"feat_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=col_names)
    df["month"] = month
    df["client_id"] = np.arange(1, n_rows + 1)
    df["default"] = default

    # Shuffle columns so default is not last if you prefer
    # Save compressed csv
    print(f"Writing {out} ... (rows={n_rows}, features={n_features})")
    df.to_csv(out, index=False, compression="gzip")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=200_000, help="Number of rows to generate")
    parser.add_argument("--n_features", type=int, default=500, help="Number of numeric features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="microloans_200k.csv.gz", help="Output filename (gzip)")
    args = parser.parse_args()
    generate_dataset(n_rows=args.n_rows, n_features=args.n_features, seed=args.seed, out=args.out)
