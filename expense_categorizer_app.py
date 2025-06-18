"""expense_categorizer_app.py – v3.4 (consolidated, tested)
================================================================
Full, self‑contained file that **retains every feature** from v3.3 plus:
• Universal NaN guard via `_prep_frame`
• Numeric‑like object columns auto‑coerced
• Date column auto‑format to MM/DD/YYYY in review grid
• Single admin user, multi‑tenant folders
• Column‑mapping UI with *Include* and *Use in Model* flags
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st  # type: ignore

# -----------------------------------------------------------------------------
# joblib import with pickle fallback (keeps Streamlit Cloud light)
# -----------------------------------------------------------------------------
try:
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import pickle

    class _PickleJoblib:
        @staticmethod
        def dump(obj, file, **kwargs):
            with open(file, "wb") if isinstance(file, (str, Path)) else file as fh:
                pickle.dump(obj, fh)

        @staticmethod
        def load(file, **kwargs):
            with open(file, "rb") if isinstance(file, (str, Path)) else file as fh:
                return pickle.load(fh)

    joblib = _PickleJoblib()  # type: ignore

# -----------------------------------------------------------------------------
# Sci‑kit Learn imports (after joblib shim)
# -----------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# Tenant filesystem helpers
# -----------------------------------------------------------------------------
BASE_DIR = Path("/mnt/data/tenants") if Path("/mnt/data").exists() else Path("tenants")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def tenant_paths(tenant: str) -> Dict[str, Path]:
    tdir = BASE_DIR / tenant
    tdir.mkdir(parents=True, exist_ok=True)
    return {
        "root": tdir,
        "model": tdir / "expense_model.joblib",
    }

# -----------------------------------------------------------------------------
# Data preparation helpers
# -----------------------------------------------------------------------------

def _prep_frame(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Return a copy with NaNs filled and numeric‑like objects coerced."""
    df = df.copy()
    # Coerce object → numeric where possible
    for col in feat_cols:
        if df[col].dtype == object:
            num = pd.to_numeric(df[col], errors="coerce")
            if num.notna().sum() > 0:  # some numeric
                df[col] = num
    # Fill NaN
    num_cols = df[feat_cols].select_dtypes("number").columns
    txt_cols = [c for c in feat_cols if c not in num_cols]
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(0)
    if txt_cols:
        df[txt_cols] = df[txt_cols].fillna("")
    return df

# -----------------------------------------------------------------------------
# Pipeline builder with empty‑text guard
# -----------------------------------------------------------------------------

def _build_pipeline(df: pd.DataFrame, feat_cols: List[str]) -> Pipeline:
    text_cols = [c for c in feat_cols if df[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in text_cols]

    valid_text: List[str] = []
    for c in text_cols:
        s = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
        if (s.str.len() > 0).any():
            valid_text.append(c)
    text_cols = valid_text

    transformers: List[Any] = [(f"tfidf_{c}", TfidfVectorizer(stop_words="english"), c) for c in text_cols]
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if not transformers:
        raise ValueError("No usable feature columns after preprocessing.")

    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))])

# -----------------------------------------------------------------------------
# Public training / prediction APIs
# -----------------------------------------------------------------------------

def train_model(df: pd.DataFrame, feat_cols: List[str], target: str = "Category") -> Pipeline:
    # Drop rows without target
    missing_target = df[target].isna() | (df[target].astype(str).str.strip() == "")
    if missing_target.any():
        st.warning(f"Dropping {missing_target.sum()} rows with blank '{target}'")
        df = df[~missing_target]
    if df.empty:
        raise ValueError("Training data has no rows with a valid Category label.")

    df = _prep_frame(df, feat_cols)
    pipe = _build_pipeline(df, feat_cols)
    pipe.fit(df[feat_cols], df[target])
    pipe.feature_columns_ = feat_cols  # type: ignore[attr-defined]
    return pipe


def predict_df(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = list(model.feature_columns_)  # type: ignore[attr-defined]
    df = _prep_frame(df, feat_cols)
    preds = model.predict(df[feat_cols])
    probs = model.predict_proba(df[feat_cols]).max(axis=1)
    df["Suggested Category"] = preds
    df["Confidence"] = probs.round(2)
    return df

# -----------------------------------------------------------------------------
# Journal Entry helper (still used downstream)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# File load helper
# -----------------------------------------------------------------------------

def _load_file(file_obj, name: str) -> pd.DataFrame:
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    if name.endswith(".json"):
        return pd.read_json(file_obj)
    raise ValueError("Unsupported file format")

# -----------------------------------------------------------------------------
# Column mapping UI
# -----------------------------------------------------------------------------

def mapping_editor(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    st.markdown("### 1A · Map & configure columns")
    cfg = pd.DataFrame({
        "Dataset Header": df.columns,
        "Program Header": df.columns,
        "Include": True,
        "Use in Model": True,
    })
    cfg.loc[cfg["Program Header"].str.lower() == "category", "Use in Model"] = False
    edited = st.data_editor(cfg, num_rows="dynamic", use_container_width=True)
    keep = edited[(edited["Include"] == True) & (edited["Program Header"].str.strip() != "")]
    col_map = dict(zip(keep["Dataset Header"], keep["Program Header"].str.strip()))
    if "Category" not in col_map.values():
        st.error("Map at least one column to `Category`.")
        return {}, []
    feat_cols = [ph for ph, use in zip(keep["Program Header"], keep["Use in Model"]) if use and ph != "Category"]
    if st.button("Confirm mapping & feature selection"):
        st.session_state["col_map"] = col_map
        st.session_state["feat_cols"] = feat_cols
        st.success("Mapping saved.")
    return st.session_state.get("col_map", {}), st.session_state.get("feat_cols", [])

# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------

def run_app():
    st.sidebar.success("Logged in as **admin** (no password)")
    tenant = st.sidebar.selectbox("Tenant", sorted([p.name for p in BASE_DIR.iterdir() if p.is_dir()]) or ["customer_one"], index=0)
    paths = tenant_paths(tenant)

    st.title(f"AI Expense Categorizer – Tenant: {tenant}")

    model_file = paths["model"]
    model: Pipeline | None = joblib.load(model_file) if model_file.exists() else None
    col_map: Dict[str, str] = st.session_state.get("col_map", {})
    feat_cols: List[str] = st.session_state.get("feat_cols", [])

    # ---------------- TRAIN -------------------
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

    # ---------------- PREDICT -----------------
    st.header("2 · Review & finalize new spend")
    if model is None or not col_map:
        st.info("Train a model first to enable prediction.")
        return

    n_file = st.file_uploader("New-period spend file", key="new")
    if n_file:
        df_new_raw = _load_file(n_file, n_file.name)
        missing = [c for c in col_map.keys() if c not in df_new_raw.columns]
        if missing:
            st.error(f"New file missing mapped columns: {', '.join(missing)}")
            return
        df_new = df_new_raw.rename(columns=col_map)

        # Format date cols to MM/DD/YYYY
        for c in df_new.columns:
            if c.lower() in {"date", "transactiondate", "transaction_date"} or pd.api.types.is_datetime64_any_dtype(df_new[c]):
                df_new[c] = pd.to_datetime(df_new[c], errors="coerce").dt.strftime("%m/%d/%Y")

        df_out = predict_df(model, df_new)
        st.dataframe(df_out, use_container_width=True)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_app()
