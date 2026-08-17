import streamlit as st
import pandas as pd

from steps.step_tuning import show_tuning_section
from steps.step1_overview import show_overview
from steps.step2_quality import show_data_quality
from steps.step3_target import select_target_and_detect
from steps.step4_preprocessing import show_preprocessing_plan
from steps.step5_visualization import show_visualizations
from steps.step6_training import train_and_compare_models
from steps.step7_prediction import show_prediction_interface
from steps.step8_confusion_matrix import show_confusion_matrix
from steps.step9_feature_importance import show_feature_importance


st.set_page_config(page_title="ML Pipeline Studio", layout="wide")

st.title("ML Pipeline Studio")
st.caption("Upload any dataset — this platform automatically analyzes it, cleans it, visualizes it, and finds the best-performing machine learning model.")

with st.expander("About This Platform", expanded=True):
    st.markdown("""
    This platform automatically analyzes any tabular dataset (CSV) and identifies the best-performing
    machine learning model — no manual coding required.

    **How it works, step by step:**
    1. Upload a CSV file
    2. The platform scans the data — missing values, duplicates, column types
    3. You select a target column (what you want to predict) — the platform automatically determines whether this is a Regression or Classification problem
    4. Data is automatically cleaned — missing values filled, categories converted to numbers, values scaled
    5. The data is visualized — distributions, outliers, and relationships between variables
    6. Nine different algorithms are trained simultaneously (Linear/Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, XGBoost, LightGBM, CatBoost, Naive Bayes)
    7. The best-performing model is automatically selected — along with the reasoning behind it

    Upload a file below to begin.
    """)

st.divider()

st.header("Step 1: Upload Your Dataset")
st.write("Start by uploading a CSV file. The platform will immediately scan it for structure, missing values, and data types.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File loaded successfully: {uploaded_file.name}")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    id_like_cols = []
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.9:
            id_like_cols.append(col)

    if id_like_cols:
        st.info(f"The following column(s) appear to be identifiers (nearly all unique values) and were automatically excluded from training as features: {', '.join(id_like_cols)}")
        numeric_cols = [c for c in numeric_cols if c not in id_like_cols]
        categorical_cols = [c for c in categorical_cols if c not in id_like_cols]
        df = df.drop(columns=id_like_cols)
    show_overview(df)
    show_data_quality(df, numeric_cols, categorical_cols)
    target_col, problem_type = select_target_and_detect(df)
    preprocessor, feature_cols = show_preprocessing_plan(df, target_col, numeric_cols, categorical_cols)
    show_visualizations(df, numeric_cols, categorical_cols, target_col)

    st.divider()

    if st.button("Train & Compare Models", type="primary", use_container_width=True):
        results = train_and_compare_models(df, target_col, problem_type, preprocessor, feature_cols)
        st.session_state["training_results"] = results
        st.session_state["problem_type"] = problem_type

    if "training_results" in st.session_state:
        show_confusion_matrix(st.session_state["training_results"], st.session_state["problem_type"])
        show_feature_importance(st.session_state["training_results"], feature_cols, numeric_cols, categorical_cols)

        valid_rows_tune = df[target_col].notna()
        X_full = df.loc[valid_rows_tune, feature_cols]
        y_full = df.loc[valid_rows_tune, target_col]
        if st.session_state["training_results"].get("label_encoder") is not None:
            y_full = st.session_state["training_results"]["label_encoder"].transform(y_full)
        show_tuning_section(st.session_state["training_results"], X_full, y_full, st.session_state["problem_type"])

    st.divider()

    if "training_results" in st.session_state:
        show_prediction_interface(
            df,
            feature_cols,
            numeric_cols,
            categorical_cols,
            st.session_state["training_results"],
            st.session_state["problem_type"]
        )

else:
    st.info("Upload a CSV file above to begin the analysis.")