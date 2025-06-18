"""expense_categorizer_app.py – v3.2 (column‑mapping UI)
===============================================================
* 🗂️ **New column‑mapping step** when you upload a training file:
  * Shows every dataset header.
  * Let’s you map each to a *Program Header* (editable) or deselect by un‑ticking *Include*.
  * Must keep (or map something to) `Category` for the target label.
* The mapping is saved in `st.session_state["col_map"]` and is automatically applied to **new‑spend** files.
* You can add new program headers simply by typing a fresh name in the *Program Header* column.

No authentication changes; still a single admin with tenant picker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st  # type: ignore

# -------------------------
# joblib shim (unchanged)
# -------------------------
try:
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
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
    }

# -------------------------
# ML helpers
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

# -------------------------
# File helpers
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

# -------------------------
# Column‑mapping UI
# -------------------------

def mapping_editor(df: pd.DataFrame) -> Dict[str, str]:
    st.markdown("### 1A · Map dataset columns → program headers")
    map_df = pd.DataFrame({
        "Dataset Header": df.columns,
        "Program Header": df.columns,
        "Include": True,
    })
    edited = st.data_editor(map_df, num_rows="dynamic", use_container_width=True)
    # Build dict of dataset→program for rows where Include True and Program Header not blank
    included = edited[(edited["Include"] == True) & (edited["Program Header"].str.strip() != "")]
    col_map = dict(zip(included["Dataset Header"], included["Program Header"].str.strip()))
    if "Category" not in col_map.values():
        st.error("You must map at least one column to `Category` – the target label.")
        return {}
    if st.button("Confirm mapping"):
        st.session_state["col_map"] = col_map
        st.success("Mapping saved – continue below.")
    return st.session_state.get("col_map", {})

# -------------------------
# Main App
# -------------------------

def run_app():
    st.sidebar.success("Logged in as **admin** (no password)")
    tenant = st.sidebar.selectbox("Tenant", sorted([p.name for p in BASE_DIR.iterdir() if p.is_dir()]) or ["customer_one"], index=0)
    paths = tenant_paths(tenant)

    st.title(f"AI Expense Categorizer – Tenant: {tenant}")

    model_file = paths["model"]
    model: Pipeline | None = joblib.load(model_file) if model_file.exists() else None
    col_map: Dict[str, str] = st.session_state.get("col_map", {})

    # ---- Train ----
    st.header("1 · Train / Retrain")
    t_file = st.file_uploader("Training data (CSV/XLSX/JSON)")
    if t_file:
        df_train_raw = _load_file(t_file, t_file.name)
        col_map = mapping_editor(df_train_raw)
        if col_map:
            df_train = df_train_raw.rename(columns=col_map)[list(col_map.values())]
            feature_cols = [c for c in df_train.columns if c != "Category"]
            if st.button("Train model", key="train_btn"):
                model = train_model(df_train, feature_cols)
                joblib.dump(model, model_file)
                st.success("Model trained & saved ✅")
                st.session_state["feature_cols"] = feature_cols
    else:
        st.info("Upload a training file to begin.")

    # ---- Predict ----
    st.header("2 · Review & finalize new spend")
    if model is None or not col_map:
        st.info("Train a model (and save column mapping) before prediction.")
        return

    n_file = st.file_uploader("New‑period spend file", key="new")
    if n_file:
        df_new_raw = _load_file(n_file, n_file.name)
        missing = [c for c in col_map.keys() if c not in df_new_raw.columns]
        if missing:
            st.error(f"New file is missing expected columns: {', '.join(missing)}")
            return
        df_new = df_new_raw.rename(columns=col_map)
        df_out = predict_df(model, df_new)
        st.dataframe(df_out, use_container_width=True)

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    run_app()
