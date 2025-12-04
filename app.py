"""
app_streamlit.py

Streamlit app for Question 2 microloan analysis.
Run: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import base64

st.set_page_config(page_title="Microloan Feature Selection & PCA", layout="wide")

st.title("Microloan Transaction Data — Feature Selection, PCA & Model Comparison")

# --- Data input ---
st.sidebar.header("Data input")
option = st.sidebar.radio("Choose data source", ["Upload CSV", "Generate synthetic (fast)", "Generate synthetic (big)"])

if option == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (gzip OK)", type=["csv", "gz", "csv.gz"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, compression='infer')
    else:
        st.stop()
elif option == "Generate synthetic (fast)":
    n_rows = st.sidebar.number_input("Rows", value=50_000, min_value=1000, step=1000)
    if st.sidebar.button("Generate now"):
        import subprocess, sys
        # generate in-memory using same logic as script
        from math import exp
        rng = np.random.default_rng(42)
        n_features = 500
        X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
        n_causal = 12
        weights = rng.normal(0.8, 0.6, size=n_causal)
        linear_score = X[:, :n_causal] @ weights
        month = rng.integers(1,13,size=n_rows)
        seasonal = np.sin(month / 12.0 * 2 * np.pi) * 0.5
        score = linear_score + seasonal + rng.normal(0, 1.2, size=n_rows)
        from scipy.special import expit
        prob = expit((score - np.mean(score))/ (np.std(score)+1e-9)) * 0.5
        default = (rng.random(n_rows) < prob).astype(int)
        cols = [f"feat_{i:03d}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["month"] = month
        df["client_id"] = np.arange(1, n_rows+1)
        df["default"] = default
    else:
        st.stop()
else:
    n_rows = st.sidebar.number_input("Rows", value=500_000, min_value=100_000, step=100_000)
    if st.sidebar.button("Generate big now"):
        rng = np.random.default_rng(42)
        n_features = 500
        X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
        n_causal = 12
        weights = rng.normal(0.8, 0.6, size=n_causal)
        linear_score = X[:, :n_causal] @ weights
        month = rng.integers(1,13,size=n_rows)
        seasonal = np.sin(month / 12.0 * 2 * np.pi) * 0.5
        score = linear_score + seasonal + rng.normal(0, 1.2, size=n_rows)
        from scipy.special import expit
        prob = expit((score - np.mean(score))/ (np.std(score)+1e-9)) * 0.5
        default = (rng.random(n_rows) < prob).astype(int)
        cols = [f"feat_{i:03d}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["month"] = month
        df["client_id"] = np.arange(1, n_rows+1)
        df["default"] = default
    else:
        st.stop()

st.write(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
st.dataframe(df.head(5))

# --- Feature selection ---
st.header("Feature selection")
target = st.selectbox("Target column", options=[c for c in df.columns if df[c].nunique() <= 100 and c.lower()!="client_id"], index=df.columns.get_loc("default") if "default" in df.columns else 0)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != target and c != "client_id"]
st.write(f"{len(feature_cols)} numeric features available")

k = st.slider("Select top-k features (by absolute Pearson correlation with target)", min_value=5, max_value=50, value=10)

# compute correlations
with st.spinner("Computing correlations..."):
    corrs = df[feature_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    topk = corrs.head(k).index.tolist()
    st.write("Top features (by abs Pearson correlation):")
    st.table(corrs.head(k))

# --- PCA ---
st.header("PCA (on selected features or all numeric)")
pca_on = st.radio("PCA input", ["Top-k features", "All numeric features"], index=0)
n_components = st.slider("PCA components", min_value=2, max_value=50, value=5)
if pca_on == "Top-k features":
    X_pca_in = df[topk].values
else:
    X_pca_in = df[feature_cols].values

with st.spinner("Scaling and fitting PCA..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca_in)
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    expl_var = pca.explained_variance_ratio_.cumsum()

st.write(f"Cumulative explained variance by {n_components} components: {expl_var[-1]:.3f}")
st.line_chart(pd.Series(pca.explained_variance_ratio_).cumsum())

# --- Model training and timing comparisons ---
st.header("Model comparison: Full features vs Top-k vs PCA")
test_size = st.slider("Test size (%)", min_value=5, max_value=40, value=20)
seed = 42

# Prepare datasets
X_full = df[feature_cols].values
y = df[target].values
X_topk = df[topk].values
X_pca_full = X_pca  # from above

# helper to train and time
def train_eval(X, y, name):
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=seed, stratify=y)
    scaler_local = StandardScaler()
    X_train = scaler_local.fit_transform(X_train)
    X_test = scaler_local.transform(X_test)
    model = LogisticRegression(max_iter=200, solver="saga")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    t1 = time.time()
    return {"name": name, "time_s": t1-t0, "auc": auc, "train_rows": X_train.shape[0], "n_features": X_train.shape[1], "size_bytes": X.nbytes}

st.write("Training (this may take some seconds)...")
res = []
res.append(train_eval(X_full, y, "Full numeric features"))
res.append(train_eval(X_topk, y, "Top-k features"))
res.append(train_eval(X_pca_full, y, f"PCA ({n_components})"))

res_df = pd.DataFrame(res).sort_values("auc", ascending=False)
st.table(res_df)

st.markdown("**Observations**")
st.write("- Full features: large number of columns → longer training time; may overfit depending on signal/noise.")
st.write("- Top-k: faster, fewer features, often similar or better generalization if signal concentrated.")
st.write("- PCA: good dimension reduction for speed; may lose interpretability.")

# --- Export reduced dataset ---
st.header("Export reduced dataset")
export_choice = st.radio("Which dataset to export", ["Top-k features + target", "PCA components + target"])
if st.button("Prepare CSV for download"):
    if export_choice.startswith("Top-k"):
        export_df = df[topk + [target]]
    else:
        pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        export_df = pd.DataFrame(X_pca, columns=pca_cols)
        export_df[target] = df[target].values
    csv_bytes = export_df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="reduced_dataset.csv">Download reduced dataset (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.write("To store in Google Drive / Google Sheets: download CSV and upload to Google Drive, then open with Google Sheets (File → Import).")

st.info("Tip: to automate uploading into Google Drive you can use Google Drive API or gspread with a service account.")
