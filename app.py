# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="💉 Diabetes Prediction ",
    layout="wide",
    page_icon="💉"
)

MODEL_PATH = "model.pkl"
DATA_PATH = "data/dataset.csv"

# ---------------- Data Loader ----------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# ---------------- Model Loader/Trainer ----------------
@st.cache_resource
def load_or_train_model(df):
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_proc = df.copy()
    for col in zero_as_missing:
        df_proc[col] = df_proc[col].replace(0, np.nan)
    imputer = SimpleImputer(strategy="median")
    df_proc[zero_as_missing] = imputer.fit_transform(df_proc[zero_as_missing])

    X, y = df_proc.drop("Outcome", axis=1), df_proc["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    log_pipe = Pipeline([
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    rf_pipe = Pipeline([
        ("scaler", scaler),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_log = cross_val_score(log_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        scores_rf = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        model = rf_pipe if scores_rf.mean() >= scores_log.mean() else log_pipe
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred)
    }
    return model, metrics, X.columns.tolist(), X_test, y_test

# ---------------- Main Title ----------------
st.markdown(
    """
    <h1 style="text-align:center; color:#FF000;">
        💉 GlucoVision- Diabetics Analyzer
    </h1>
    <h6 style="text-align:center; color:gray;">
       <P> GlucoVision is an AI-powered diabetes risk prediction and monitoring 
       application designed to help users take proactive control of their health. 
       <br>By analyzing personal health metrics such as blood glucose levels,BMI level, etc, GlucoVision provides accurate predictions, early warnings.
    </h6>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar Navigation ----------------
with st.sidebar:
    st.header("📌 Navigation")
    page = st.radio("", ["📊 Data Explorer", "📈 Visualisations", "📟 Model Dashboard", "🤖 Predict"])
    uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

df = uploaded and pd.read_csv(uploaded) or load_data(DATA_PATH)
model, metrics, feature_names, X_test, y_test = load_or_train_model(df)

# ---------------- Page: Data Explorer ----------------
if page == "📊 Data Explorer":
    st.subheader("🔍 Dataset Overview")
    st.dataframe(df.head())
    st.success(f"✅ Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe().T.style.background_gradient(cmap="Blues").format(precision=2))

# ---------------- Page: Visualisations ----------------
elif page == "📈 Visualisations":
    st.subheader("📊 Interactive Charts")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    sel = st.selectbox("📌 Select Numeric Column", num_cols)
    st.plotly_chart(px.histogram(df, x=sel, marginal="box", nbins=30, color_discrete_sequence=["#1f77b4"]))

    st.subheader("🔗 Correlation Heatmap")
    corr = df[num_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis"))

# ---------------- Page: Model Dashboard ----------------
elif page == "📟 Model Dashboard":
    st.subheader("📊 Model Performance Meters")

    cols = st.columns(5)
    for idx, (label, val) in enumerate({
        "🎯 Accuracy": metrics["accuracy"],
        "📏 Precision": metrics["precision"],
        "🔍 Recall": metrics["recall"],
        "⚖️ F1": metrics["f1"],
        "📈 ROC AUC": metrics["roc_auc"]
    }.items()):
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val*100,
            title={"text": label, "font": {"size": 16}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FF4B4B"}}
        ))
        cols[idx].plotly_chart(gauge, use_container_width=True)

    # New: Class Distribution
    st.subheader("📊 Class Distribution")
    pie_fig = px.pie(
        df, names="Outcome",
        title="Positive vs Negative Outcomes",
        color="Outcome",
        color_discrete_map={0: "#1f77b4", 1: "#ff7f0e"},
        hole=0.4
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # New: Feature Importance
    st.subheader("🔥 Feature Importance")
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf_model = model.named_steps["clf"]
        if hasattr(clf_model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": clf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
        elif hasattr(clf_model, "coef_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": np.abs(clf_model.coef_[0])
            }).sort_values(by="Importance", ascending=False)
        else:
            importance_df = None

        if importance_df is not None:
            bar_fig = px.bar(
                importance_df,
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("⚠️ Feature importance is not available for this model.")

# ---------------- Page: Predict ----------------
elif page == "🤖 Predict":
    st.subheader("🧮 Enter Feature Values")
    user_data = {}
    for f in feature_names:
        user_data[f] = st.number_input(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
    if st.button("🚀 Predict"):
        X_new = pd.DataFrame([user_data])
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]
        st.markdown(f"### 🩺 Prediction: **{'Positive' if pred==1 else 'Negative'}**")
        st.progress(int(prob*100))
        st.info(f"🔢 Probability of Positive: {prob:.2%}")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p><b>Developed by Sandun Indrasiri Wijesingha</b></p>
        <p>📧 Contact: <a href='mailto:ssandu809@gmail.com'>ssandu809@gmail.com</a></p>
        <p>© All Rights Reserved®️</p>
    </div>
    """,
    unsafe_allow_html=True
)
