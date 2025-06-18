"""expense_categorizer_app.py – v3.0 (multi‑tenant)
=======================================================
Adds a **basic admin console + multi‑tenant support** on top of the v2.3
functionality.  Key features:

* 🔐 **Username/password authentication** via *streamlit‑authenticator*.
* 🏢 **Tenants** – each customer works inside their own folder under
  `/mnt/data/tenants/<tenant>/` (models, uploads, logos).
* 🖼️ **Per‑tenant logo** – customers can upload a PNG/JPG which shows in the
  sidebar on later visits.
* 👩‍💼 **Developer role** – a special *developer* user can switch between
  tenants using a selectbox in the sidebar and view/manage any instance.

> **NOTE:** to keep the example self‑contained we embed an in‑memory user list
> and hashed passwords.  Swap for a DB or SSO provider in production.
"""
from __future__ import annotations

import sys as _sys
import types as _types
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    raise RuntimeError("Streamlit not installed – `pip install streamlit`.")

try:
    import streamlit_authenticator as stauth  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    raise RuntimeError("Missing dependency: `pip install streamlit‑authenticator`. ")

# ------------------------------------------------------------------
# Simple user store – replace with DB/SSO in prod
# ------------------------------------------------------------------
hasher = stauth.Hasher(["custpw", "devpw"]).generate()
USERS = {
    "customer": {
        "name": "Customer One",
        "password": hasher[0],
        "email": "cust@example.com",
        "role": "customer",
        "tenant": "customer_one",
    },
    "developer": {
        "name": "Developer",
        "password": hasher[1],
        "email": "dev@example.com",
        "role": "developer",
        "tenant": "*",  # wildcard
    },
}

# ------------------------------------------------------------------
# Tenant paths / helpers
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Core ML helpers (unchanged logic)
# ------------------------------------------------------------------

def _build_pipeline(df: pd.DataFrame, feat_cols: List[str]) -> Pipeline:
    text_cols = [c for c in feat_cols if df[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in text_cols]
    transformers = [(f"tfidf_{c}", TfidfVectorizer(stop_words="english"), c) for c in text_cols]
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))])


def train_model(df: pd.DataFrame, feat_cols: List[str], target: str = "Category") -> Pipeline:
    df = df.copy()
    df[feat_cols] = df[feat_cols].fillna("")
    pipe = _build_pipeline(df, feat_cols)
    pipe.fit(df[feat_cols], df[target])
    pipe.feature_columns_ = feat_cols  # type: ignore[attr-defined]
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

# ------------------------------------------------------------------
# Utility – load files / editable grid
# ------------------------------------------------------------------

def _load_file(file_obj, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    if name.endswith(".json"):
        return pd.read_json(file_obj)
    raise ValueError("Unsupported file – CSV, XLSX, or JSON only")


def _edit(df: pd.DataFrame) -> pd.DataFrame:
    return (st.data_editor if hasattr(st, "data_editor") else st.experimental_data_editor)(
        df, num_rows="dynamic", use_container_width=True  # type: ignore[attr-defined]
    )

# ------------------------------------------------------------------
# Authentication + Tenant selection UI
# ------------------------------------------------------------------

def authenticate() -> tuple[bool, str, Dict[str, Any]]:
    names = {u: USERS[u]["name"] for u in USERS}
    pwds = {u: USERS[u]["password"] for u in USERS}
    cookies = {
        "expiry_days": 7,
        "key": "ec_auth",
        "name": "ec_auth_cookie",
    }
    authenticator = stauth.Authenticate(names, pwds, cookies, "eccat_v3")

    name, auth_status, username = authenticator.login("Login", "main")
    if not auth_status:
        st.warning("Please enter your credentials." if auth_status is None else "Invalid credentials.")
        st.stop()
    user_info = USERS[username]
    st.sidebar.success(f"Logged in as {user_info['name']}")
    if user_info["role"] == "developer":
        tenants = [p.name for p in BASE_DIR.iterdir() if p.is_dir()] or ["customer_one"]
        tenant = st.sidebar.selectbox("Select tenant", tenants)
    else:
        tenant = user_info["tenant"]
    st.sidebar.caption(f"Active tenant: **{tenant}**")
    return True, tenant, user_info

# ------------------------------------------------------------------
# Main Streamlit App
# ------------------------------------------------------------------

def run_app():
    authed, tenant, user = authenticate()
    paths = tenant_paths(tenant)

    # Sidebar logo upload / display
    if paths["logos"].exists():
        st.sidebar.image(str(paths["logos"]), width=160)
    if user["role"] == "customer":
        logo_file = st.sidebar.file_uploader("Upload logo", type=["png", "jpg", "jpeg"], key="logo")
        if logo_file and st.sidebar.button("Save logo"):
            paths["logos"].write_bytes(logo_file.read())
            st.sidebar.success("Logo saved – refresh to view")

    # Load model if exists
    model_file = paths["model"]
    model: Pipeline | None = joblib.load(model_file) if model_file.exists() else None

    st.title("AI Expense Categorizer – Tenant: " + tenant)

    # ----------------- Training -----------------
    st.header("1 · Train / Retrain")
    t_file = st.file_uploader("Training CSV/XLSX/JSON", key="train")
    if t_file:
        try:
            df_train = _load_file(t_file, t_file.name)
            st.success(f"Loaded {len(df_train):,} rows")
            default_feats = [c for c in df_train.columns if c != "Category"]
            feats = st.multiselect("Feature columns", df_train.columns, default=default_feats)
            if st.button("Train model"):
                mdl = train_model(df_train, feats)
                joblib.dump(mdl, model_file)
                st.success("Model trained & saved")
                model = mdl
                st.session_state["feature_cols"] = feats
        except Exception as e:
            st.error(str(e))

    # ----------------- Prediction ---------------
    st.header("2 · Review new spend → finalize & retrain")
    if model is None:
        st.info("Upload training data first.")
        st.stop()
    feats = st.session_state.get("feature_cols", list(getattr(model, "feature_columns_", [])))

    n_file = st.file_uploader("New spend", key="new")
    posted_all = st.checkbox("All rows posted to GL", value=False)
    posted_col = None if posted_all else st.text_input("Posted flag column (optional)")

    map_file = st.file_uploader("Category→Account map CSV", key="map")
    suspense = st.text_input("Suspense account", value="9999")

    if n_file:
        try:
            df_new = _load_file(n_file, n_file.name)
            df_out = predict_df(model, df_new)
            if posted_all:
                df_out["Posted"] = True
            elif posted_col and posted_col in df_out.columns:
                df_out.rename(columns={posted_col: "Posted"}, inplace=True)
            else:
                df_out["Posted"] = False
            st.markdown("#### Edit and finalize")
            edited_df = _edit(df_out)
            if st.button("Finalize & retrain", type="primary"):
                finalized = edited_df.copy()
                finalized["Category"] = finalized["Suggested Category"]
                mdl = train_model(finalized, feats)
                joblib.dump(mdl, model_file)
                model = mdl
                st.success("Model retrained on finalized data")
            st.download_button("Download corrected CSV", edited_df.to_csv(index=False).encode(),
                               "corrected_expenses.csv")
            # JEs
            if map_file and edited_df["Posted"].any():
                mdf = pd.read_csv(map_file)
                acct_map = dict(zip(mdf.Category.astype(str), mdf.Account.astype(str)))
                je_df = generate_journal_entries(edited_df[edited_df.Posted], acct_map, suspense)
                st.subheader("Suggested JEs")
                st.dataframe(je_df, use_container_width=True)
                st.download_button("Download JE CSV", je_df.to_csv(index=False).encode(),
                                   "reclass_journal_entries.csv")
        except Exception as e:
            st.error(str(e))

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_app()
