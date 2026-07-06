# Interpretable ML for Breast Cancer Diagnosis

Random Forest classifier with SHAP-based explainability, deployed as an
interactive Streamlit application.

**Live demo:** https://breast-cancer-interpretable-ml.streamlit.app

## Overview

A clinical ML model is only useful if a clinician can see *why* it made a
prediction. This project trains a Random Forest on the Wisconsin Diagnostic
Breast Cancer dataset (569 samples, 30 features) and pairs it with SHAP
(SHapley Additive exPlanations) to produce:

- **Global explanations** — which features drive the model across all patients
- **Local explanations** — why the model classified one specific patient
  as malignant or benign, shown as a per-prediction waterfall

## Results

| Model | ROC-AUC (test) |
|---|---|
| Logistic Regression | 0.995 |
| Random Forest | 0.993 |

**Top predictive features by SHAP (Random Forest):**
worst perimeter, worst area, worst concave points, worst radius,
mean concave points.

In the SHAP summary plot, high values of these features (large tumor
measurements) push predictions away from *benign* toward *malignant* —
the clinically expected direction, and a basic sanity check that the
model learned something real rather than an artifact.

![SHAP summary plot](plots/shap_summary.png)

## A finding worth noting: interpretability methods disagree

SHAP (TreeExplainer) and permutation importance, run on the *same*
Random Forest, rank features differently. SHAP puts *worst perimeter* and
*worst area* at the top; permutation importance surfaces a different
ordering. This isn't a bug — the two methods measure different things
(a feature's marginal contribution to each prediction vs. the drop in
model performance when that feature is shuffled). It raises a real
question for clinical deployment: **when two accepted attribution methods
disagree, which should a clinician trust?** That question is the
direction this project extends into.

## Interpretability methods compared

- **Permutation importance** — global, model-agnostic feature ranking
- **SHAP (TreeExplainer)** — per-prediction attributions with direction
  of effect, grounded in cooperative game theory

## Run locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Repo structure

\`\`\`
├── app.py                                  # Streamlit application
├── Breast_Cancer_Interpretable_ML.ipynb    # full analysis notebook
├── requirements.txt
├── runtime.txt                             # pins Python 3.11 for SHAP
├── plots/                                  # exported SHAP figures
│   ├── shap_summary.png
│   └── shap_waterfall_sample.png
└── README.md
\`\`\`

## Limitations

Single curated dataset; classical models only. Planned extension:
multimodal fusion with ClinicalBERT for unstructured clinical text,
carrying the same interpretability question into a transformer setting.

## Disclaimer

Educational project. Not a medical device; not for diagnostic use.
