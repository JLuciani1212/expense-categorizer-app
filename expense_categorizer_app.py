"""expense_categorizer_app.py – v3.3 (column include/use‑in‑model flags)
===================================================================
* Adds **Use in Model** flag next to each mapped column so a field can appear
  in the review grid but be excluded from training/prediction.
* Blank/NaN fields are acceptable; skew columns can simply be marked *Use in
  Model = False*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st  # type: ignore

# ------------------------------------------------------------------
# joblib shim (unchanged)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Tenant filesystem helpers
# ------------------------------------------------------------------
BASE_DIR = Path("/mnt/data/tenants") if Path("/mnt/data").exists() else Path("tenants")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def tenant_paths(tenant: str) -> Dict[str, Path]:
    tdir = BASE_DIR / tenant
    tdir.mkdir(parents=True, exist_ok=True)
    return {"root": tdir, "model": tdir / "expense_model.joblib"}

# ------------------------------------------------------------------
# ML helpers with empty-text guard
# ------------------------------------------------------------------

def _build_pipeline(df: pd.DataFrame, feat_cols: List[str]) -> Pipeline:
    text_cols = [c for c in feat_cols if df[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in text_cols]

    valid_text: List[str] = []
    for col in text_cols:
        s = df[col].astype(str).str.strip().replace({"nan": "", "None": ""})
        if (s.str.len() > 0).any():
            valid_text.append(col)
    text_cols = valid_text

    transformers: List[Any] = [(f"tfidf_{c}", TfidfVectorizer(stop_words="english"), c) for c in text_cols]
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if not transformers:
        raise ValueError("No usable feature columns after preprocessing.")

    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))])


def train_model(df: pd.DataFrame, feat_cols: List[str], target: str = "Category") -> Pipeline:
    df = df.copy()
    text_cols = [c for c in feat_cols if df[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in text_cols]
    if text_cols:
        df[text_cols] = df[text_cols].fillna("")
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)
    pipe = _build_pipeline(df, feat_cols)
    pipe.fit(df[feat_cols], df[target])
    pipe.feature_columns_ = feat_cols  # type: ignore[attr-defined]
    return pipe


def predict_df(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = [c for c in model.feature_columns_ if df[c].dtype == "object"]  # type: ignore[attr-defined]
    num_cols = [c for c in model.feature_columns_ if c not in text_cols]  # type: ignore[attr-defined]
    if text_cols:
        df[text_cols] = df[text_cols].fillna("")
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)
    preds = model.predict(df[model.feature_columns_])  # type: ignore[attr-defined]
    probs = model.predict_proba(df[model.feature_columns_]).max(axis=1)  # type: ignore[attr-defined]
    df["Suggested Category"] = preds
    df["Confidence"] = probs.round(2)
    return df

# ------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------

def _load_file(file_obj, name: str) -> pd.DataFrame:
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    if name.endswith(".json"):
        return pd.read_json(file_obj)
    raise ValueError("Unsupported file format")

# ------------------------------------------------------------------
# Column mapping UI
# ------------------------------------------------------------------

def mapping_editor(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """Return column‑rename mapping and list of columns to use in model."""
    st.markdown("### 1A · Map & configure columns")
    init_prog_headers = df.columns
    map_df = pd.DataFrame({
        "Dataset Header": df.columns,
        "Program Header": init_prog_headers,
        "Include": True,
        "Use in Model": True,
    })
    # Auto‑disable "Use in Model" for Category guess
    cat_guess_idx = map_df["Program Header"].str.lower() == "category"
    map_df.loc[cat_guess_idx, "Use in Model"] = False

    edited = st.data_editor(map_df, num_rows="dynamic", use_container_width=True)
    keep = edited[(edited["Include"] == True) & (edited["Program Header"].str.strip() != "")]
    col_map = dict(zip(keep["Dataset Header"], keep["Program Header"].str.strip()))

    if "Category" not in col_map.values():
        st.error("Map at least one column to `Category` (target label)")
        return {}, []

    feat_cols = [hdr for hdr, use in zip(keep["Program Header"], keep["Use in Model"]) if use and hdr != "Category"]

    if not feat_cols:
        st.warning("No columns selected for model features. You can still continue but model will not train.")

    if st.button("Confirm mapping & feature selection"):
        st.session_state["col_map"] = col_map
        st.session_state["feat_cols"] = feat_cols
        st.success("Settings saved – continue below.")
    return st.session_state.get("col_map", {}), st.session_state.get("feat_cols", [])

# ------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------

def run_app():
    st.sidebar.success("Logged in as **admin** (no password)")
    tenant = st.sidebar.selectbox("Tenant", sorted([p.name for p in BASE_DIR.iterdir() if p.is_dir()]) or ["customer_one"], index=0)
    paths = tenant_paths(tenant)
    st.title(f"AI Expense Categorizer – Tenant: {tenant}")

    model_file = paths["model"]
    model: Pipeline | None = joblib.load(model_file) if model_file.exists() else None
    col_map: Dict[str, str] = st.session_state.get("col_map", {})
    feat_cols: List[str] = st.session_state.get("feat_cols", [])

    # ---- Train ----
    st.header("1 · Train / Retrain")
    t_file = st.file_uploader("Training data (CSV/XLSX/JSON)")
    if t_file:
        df_raw = _load_file(t_file, t_file.name)
        col_map, feat_cols = mapping_editor(df_raw)
        if col_map and feat_cols:
            df_train = df_raw.rename(columns=col_map)[list(col_map.values())]
            if st.button("Train model"):
                try:
                    model = train_model(df_train, feat_cols)
                    joblib.dump(model, model_file)
                    st.success("Model trained & saved ✅")
                except ValueError as e:
                    st.error(str(e))
    else:
        st.info("Upload a training file to begin.")

    # ---- Predict ----
    st.header("2 · Review & finalize new spend")
    if model is None or not col_map:
        st.info("Train a model first.")
        return

    n_file = st.file_uploader("New‑period spend file", key="new")
    if n_file:
        df_new_raw = _load_file(n_file, n_file.name)
        missing = [c for c in col_map.keys() if c not in df_new_raw.columns]
        if missing:
            st.error(f"New file missing columns: {', '.join(missing)}")
            return
        df_new = df_new_raw.rename(columns=col_map)

        # ---- Date formatting to MM/DD/YYYY ---------------------------------
        for col in df_new.columns:
            if col.lower() in {"date", "transactiondate", "transaction_date"} or pd.api.types.is_datetime64_any_dtype(df_new[col]):
                try:
                    df_new[col] = pd.to_datetime(df_new[col], errors="coerce").dt.strftime("%m/%d/%Y")
                except Exception:
                    # leave as is if conversion fails
                    pass
        # -------------------------------------------------------------------

        df_out = predict_df(model, df_new)
        st.dataframe(df_out, use_container_width=True)(df_out, use_container_width=True)

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_app()
