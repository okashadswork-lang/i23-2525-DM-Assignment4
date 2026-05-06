import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

INPUT_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
FEATURE_HINTS = {
    "age": "20-80",
    "sex": "0=female, 1=male",
    "cp": "0-3",
    "trestbps": "80-200",
    "chol": "100-600",
    "fbs": "0 or 1",
    "restecg": "0-2",
    "thalach": "70-210",
    "exang": "0 or 1",
    "oldpeak": "0.0-6.0",
    "slope": "0-2",
    "ca": "0-3",
    "thal": "1-3",
}

@st.cache_resource
def load_artifacts():
    # compatibility shim for sklearn pickles referencing an older
    # private class name (_RemainderColsList). Some saved preprocessors
    # were pickled with older scikit-learn versions that defined this
    # helper. Ensure the symbol exists so joblib can unpickle the object.
    try:
        import sklearn.compose._column_transformer as _ct

        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        # If sklearn isn't available or the module layout is different,
        # proceed and let joblib raise the original error.
        pass
    preprocessor = joblib.load(ARTIFACT_DIR / "heart_preprocessor.joblib")
    model = xgb.XGBClassifier()
    model.load_model(str(ARTIFACT_DIR / "heart_xgboost.json"))
    metadata = json.loads((ARTIFACT_DIR / "heart_metadata.json").read_text())
    return preprocessor, model, metadata


def make_input_form(default_row):
    cols = st.columns(2)
    values = {}
    for idx, feature in enumerate(INPUT_ORDER):
        with cols[idx % 2]:
            values[feature] = st.number_input(
                f"{feature} ({FEATURE_HINTS[feature]})",
                value=float(default_row.get(feature, 0)),
                key=feature,
            )
    return pd.DataFrame([values])


def risk_label(probability):
    if probability >= 0.5:
        return "Disease Present", "#b91c1c"
    return "No Disease", "#15803d"


def build_explanation(top_features):
    names = ", ".join(top_features[:3])
    return f"The model is mainly driven by {names}. Review these findings together with the patient's symptoms, ECG, and overall clinical picture before deciding on follow-up."


def main():
    st.set_page_config(page_title="Heart Disease Screening", layout="wide")
    st.title("CardioAI Heart Disease Screening")
    st.write("Local-only dashboard for a saved heart-disease model.")

    preprocessor, model, metadata = load_artifacts()
    default_row = metadata.get("sample_patient", {})
    input_df = make_input_form(default_row)

    if st.button("Predict"):
        # Ensure input_df has all columns in the right order
        input_df_aligned = input_df[INPUT_ORDER]
        transformed = preprocessor.transform(input_df_aligned)
        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)
        
        # Reorder to match expected feature names from metadata
        expected_features = metadata.get("feature_names", list(feature_names))
        # Select only features that exist in transformed_df and match expected order
        available_features = [col for col in expected_features if col in transformed_df.columns]
        
        # If we're missing remainder columns, add them with their original values
        if len(available_features) < len(expected_features):
            # Add remainder columns if missing
            for col in expected_features:
                if col not in transformed_df.columns:
                    if col == "remainder__sex":
                        transformed_df[col] = input_df_aligned["sex"].values
                    elif col == "remainder__fbs":
                        transformed_df[col] = input_df_aligned["fbs"].values
                    elif col == "remainder__exang":
                        transformed_df[col] = input_df_aligned["exang"].values
            available_features = expected_features
        
        transformed_df = transformed_df[available_features]
        probability = float(model.predict_proba(transformed_df)[:, 1][0])
        prediction = int(probability >= 0.5)
        label_text, label_color = risk_label(probability)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown(f"<div style='padding:14px;border-radius:12px;background:{label_color};color:white;font-size:20px;font-weight:700;'>Prediction: {label_text}</div>", unsafe_allow_html=True)
            st.metric("Confidence", f"{probability*100:.1f}%")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        top_idx = np.argsort(np.abs(shap_values[0]))[::-1][:3]
        top_features = list(transformed_df.columns[top_idx])
        top_scores = np.abs(shap_values[0][top_idx])

        with col2:
            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.barh(top_features[::-1], top_scores[::-1], color="#ef4444")
            ax.set_title("Top 3 drivers")
            st.pyplot(fig)

        st.write(build_explanation(top_features))
        st.caption(f"Predicted class: {prediction} | Probability of disease: {probability:.3f}")


if __name__ == "__main__":
    main()
