"""expense_categorizer_app.py – v3.1 (simplified auth)
==============================================================
**Change log**
* 🔓 Removed `streamlit-authenticator` entirely.
* Single hard‑coded user (“admin”) can pick any tenant from a dropdown.
* No passwords for now; revisit multi‑user security later.
* All core ML / tenant logic untouched.

This eliminates the Hasher API headaches and gets you unblocked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import sys as _sys
import pandas as pd

# Fallback joblib shim (unchanged)
try:
    import joblib  # type: ignore
except ModuleNotFoundError:
    import pickle

    class _PickleShim:
        @staticmethod
        def dump(obj, file, **kwargs):
            with open(file, "wb") if isinstance(file, (str, Path)) else file as fh:
                pickle.dump(obj, fh)

        @staticmethod
        def load(file, **kwargs):
            with open(file, "rb") if isinstance(file, (str, Path)) else file as fh:
                return pickle.load(fh)

    joblib = _PickleShim()  # type: ignore

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import streamlit as st  # type: ignore

# -------------------------
# Simple admin user (no pw)
# -------------------------
ADMIN_USER: Dict[str, Any] = {
    "username": "admin",
    "role": "admin",
}

# -------------------------
# Tenant filesystem
# -------------------------
BASE_DIR = Path("/mnt/data/tenants") if Path("/mnt/data").exists() else Path("tenants")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def tenant_paths(tenant: str) -> Dict[str, Path]:
    tdir = BASE_DIR / tenant
    tdir.mkdir(parents=True, exist_ok=True)
    return {
        "root": tdir,
        "model": tdir / "expense_model.joblib",
        "logos": tdir / "logo.png",
    }

# -------------------------
# ML helpers (unchanged)
# -------------------------

def _build_pipeline(df: pd.DataFrame, cols: List[str]) -> Pipeline:
    text_cols = [c for c in cols if df[c].dtype == "object"]
    num_cols = [c for c in cols if c not in text_cols]
    transformers = [(f"tfidf_{c}", TfidfVectorizer(stop_words="english"), c) for c in text_cols]
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))])


def train_model(df: pd.DataFrame, cols: List[str], target: str = "Category") -> Pipeline:
    df = df.copy()
    df[cols] = df[cols].fillna("")
    pipe = _build_pipeline(df, cols)
    pipe.fit(df[cols], df[target])
    pipe.feature_columns_ = cols  # type: ignore[attr-defined]
    return pipe


def predict_df(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[model.feature_columns_] = df[model.feature_columns_].fillna("")  # type: ignore[attr-defined]
    preds = model.predict(df[model.feature_columns_])  # type: ignore[attr-defined]
    probs = model.predict_proba(df[model.feature_columns_]).max(axis=1)  # type: ignore[attr-defined]
    df["Suggested Category"] = preds
    df["Confidence"] = probs.round(2)
    return df


def generate_journal_entries(df: pd.DataFrame, acct_map: Dict[str, str], suspense: str = "9999",
                              date_col: str | None = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        cat = r["Suggested Category"]
        rows.append({"Date": r.get(date_col, "") if date_col else "", "Account": acct_map.get(cat, "UNK"),
                     "Debit": r["Amount"], "Credit": 0, "Memo": f"Reclass to {cat}"})
        rows.append({"Date": r.get(date_col, "") if date_col else "", "Account": suspense,
                     "Debit": 0, "Credit": r["Amount"], "Memo": f"Reclass to {cat}"})
    return pd.DataFrame(rows)

# -------------------------
# Utility helpers
# -------------------------

def _load_file(file_obj, name: str) -> pd.DataFrame:
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    if name.endswith(".json"):
        return pd.read_json(file_obj)
    raise ValueError("Unsupported file format")


def _edit(df: pd.DataFrame) -> pd.DataFrame:
    editor = st.data_editor if hasattr(st, "data_editor") else st.experimental_data_editor  # type: ignore[attr-defined]
    return editor(df, num_rows="dynamic", use_container_width=True)

# -------------------------
# Main App
# -------------------------

def run_app():
    st.sidebar.success("Logged in as admin (no password)")
    # Tenant picker
    tenants = sorted([p.name for p in BASE_DIR.iterdir() if p.is_dir()]) or ["customer_one"]
    tenant = st.sidebar.selectbox("Select tenant", tenants, index=0)
    paths = tenant_paths(tenant)

    # Optional logo upload
    if paths["logos"].exists():
        st.sidebar.image(str(paths["logos"]), width=160)
    logo = st.sidebar.file_uploader("Upload logo", ["png", "jpg", "jpeg"], key="logo")
    if logo and st.sidebar.button("Save logo"):
        paths["logos"].write_bytes(logo.read())
        st.sidebar.success("Logo saved – reload")

    st.title(f"AI Expense Categorizer – Tenant: {tenant}")

    model_file = paths["model"]
    model: Pipeline | None = joblib.load(model_file) if model_file.exists() else None

    # ---- Train ----
    st.header("1 · Train / Retrain")
    t_file = st.file_uploader("Training data", key="train")
    if t_file:
        df_train = _load_file(t_file, t_file.name)
        default_cols = [c for c in df_train.columns if c != "Category"]
        feat_cols = st.multiselect("Feature columns", df_train.columns, default=default_cols)
        if st.button("Train model"):
            model = train_model(df_train, feat_cols)
            joblib.dump(model, model_file)
            st.success("Model saved")
            st.session_state["feature_cols"] = feat_cols

    # ---- Predict ----
    if model is None:
        st.info("Train a model first.")
        return

    st.header("2 · Review & finalize new spend")
    n_file = st.file_uploader("New spend", key="new")
    posted_all = st.checkbox("All rows posted to GL", False)
    posted_col = st.text_input("Posted flag column", "") if not posted_all else None
    map_file = st.file_uploader("Category→Account map", key="map")
    suspense = st.text_input("Suspense account", "9999")

    if n_file:
        df_new = _load_file(n_file, n_file.name)
        df_out = predict_df(model, df_new)
        if posted_all:
            df_out["Posted"] = True
        elif posted_col and posted_col in df_out.columns:
            df_out.rename(columns={posted_col: "Posted"}, inplace=True)
        else:
            df_out["Posted"] = False
        st.markdown("#### Edit below then finalize")
        edited = _edit(df_out)
        if st.button("Finalize & retrain", type="primary"):
            edited["Category"] = edited["Suggested Category"]
            feat_cols = st.session_state.get("feature_cols", list(getattr(model, "feature_columns_", [])))
            model = train_model(edited, feat_cols)
            joblib.dump(model, model_file)
            st.success("Model retrained")
        st.download_button("Download corrected CSV", edited.to_csv(index=False).encode(), "corrected_expenses.csv")
        if map_file and edited["Posted"].any():
            mdf = pd.read_csv(map_file)
            acct_map = dict(zip(mdf.Category.astype(str), mdf.Account.astype(str)))
            je_df = generate_journal_entries(edited[edited.Posted], acct_map, suspense)
            st.subheader("Suggested JEs")
            st.dataframe(je_df, use_container_width=True)
            st.download_button("Download JE CSV", je_df.to_csv(index=False).encode(), "reclass_journal_entries.csv")

# -------------------------
# Entry‑point – Streamlit Cloud
# -------------------------
if __name__ == "__main__":
    run_app()
