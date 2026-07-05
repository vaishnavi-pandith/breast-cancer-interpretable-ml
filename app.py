"""
Interpretable ML for Breast Cancer Diagnosis
Random Forest + SHAP + Streamlit
Author: Vaishnavi Pandith
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Interpretable Breast Cancer Diagnosis",
                   layout="wide")


# ---------- Model training (cached, runs once per deployment) ----------
@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    explainer = shap.TreeExplainer(rf)
    shap_test = explainer(X_test)
    if len(shap_test.shape) == 3:
        shap_test = shap_test[:, :, 1]

    return data, X, y, X_train, X_test, y_test, rf, auc, explainer, shap_test


data, X, y, X_train, X_test, y_test, rf, auc, explainer, shap_test = load_and_train()

st.title("Interpretable ML for Breast Cancer Diagnosis")
st.caption(
    "Random Forest classifier on the Wisconsin Diagnostic dataset "
    f"(569 samples, 30 features) — test ROC-AUC: **{auc:.3f}**. "
    "Every prediction is explained with SHAP."
)
st.warning("Educational demo — not a medical device. "
           "Not for real diagnostic use.")

tab1, tab2, tab3 = st.tabs(
    ["🔍 Predict & Explain", "🌍 Global Explainability", "📊 Model Performance"]
)

# ---------- Tab 1: individual prediction + SHAP waterfall ----------
with tab1:
    st.subheader("Enter patient measurements")
    st.caption("Defaults are dataset means. Adjust the top features "
               "in the sidebar to see the prediction and its explanation change.")

    # Top features get sliders in the sidebar; the rest stay at their means.
    top_features = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )

    input_values = X.mean().copy()
    st.sidebar.header("Adjust key features")
    for feat in top_features:
        input_values[feat] = st.sidebar.slider(
            feat,
            float(X[feat].min()),
            float(X[feat].max()),
            float(X[feat].mean()),
        )

    input_df = pd.DataFrame([input_values])
    proba = rf.predict_proba(input_df)[0]
    pred_class = data.target_names[int(np.argmax(proba))]

    col1, col2 = st.columns(2)
    col1.metric("Prediction", pred_class.capitalize())
    col2.metric("Confidence", f"{proba.max():.1%}")

    st.subheader("Why did the model predict this?")
    shap_input = explainer(input_df)
    if len(shap_input.shape) == 3:
        shap_input = shap_input[:, :, 1]

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_input[0], max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption(
        "Red bars push the prediction toward class 1 (benign), "
        "blue toward class 0 (malignant). The plot decomposes this "
        "specific prediction into per-feature contributions."
    )

# ---------- Tab 2: global SHAP ----------
with tab2:
    st.subheader("Which features drive the model overall?")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_test.values, X_test, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.subheader("Mean |SHAP| value per feature")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_test.values, X_test, plot_type="bar", show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

# ---------- Tab 3: performance ----------
with tab3:
    st.subheader("Held-out test performance")
    st.metric("ROC-AUC (test set)", f"{auc:.4f}")

    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(
        cmap="Blues", ax=ax3
    )
    st.pyplot(fig3, clear_figure=True)
    st.caption(
        "In a clinical context, false negatives (missed malignancies) "
        "are the costliest error — the confusion matrix makes that "
        "trade-off visible instead of hiding it behind a single accuracy number."
    )
