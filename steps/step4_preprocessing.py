import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def show_preprocessing_plan(df, target_col, numeric_cols, categorical_cols):
    """Step: Explain and build the preprocessing pipeline for feature columns."""
    st.header("Step 5: Preprocessing Plan")
    st.write("Before training, the platform automatically prepares the data. Below is the exact transformation plan — not just what will happen, but the actual values that will be used.")

    feature_cols = [c for c in df.columns if c != target_col]
    feature_numeric = [c for c in feature_cols if c in numeric_cols]
    feature_categorical = [c for c in feature_cols if c in categorical_cols]

    if feature_numeric:
        st.subheader("Numeric Columns — Detailed Plan")
        rows = []
        for col in feature_numeric:
            missing_count = df[col].isnull().sum()
            median_val = df[col].median()
            mean_val = df[col].mean()
            std_val = df[col].std()
            rows.append({
                "Column": col,
                "Missing Values": int(missing_count),
                "Fill Value (Median)": round(median_val, 2) if pd.notna(median_val) else "N/A",
                "Mean (before scaling)": round(mean_val, 2) if pd.notna(mean_val) else "N/A",
                "Std Dev (before scaling)": round(std_val, 2) if pd.notna(std_val) else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "Wherever a value is missing, the 'Fill Value (Median)' shown above will be used to fill it. "
            "After filling, every column is scaled using StandardScaler, which applies this formula to every value: "
            "(value - mean) / std_dev. This brings every numeric column onto the same scale (centered around 0), so no single "
            "column dominates the model just because its raw numbers happen to be larger."
        )

    if feature_categorical:
        st.subheader("Categorical Columns — Detailed Plan")
        rows = []
        for col in feature_categorical:
            missing_count = df[col].isnull().sum()
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
            unique_count = df[col].nunique()
            rows.append({
                "Column": col,
                "Missing Values": int(missing_count),
                "Fill Value (Most Frequent)": mode_val,
                "Unique Categories": unique_count,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "Wherever a value is missing, the 'Fill Value (Most Frequent)' shown above will be used to fill it. "
            "After filling, each category is converted to numbers using One-Hot Encoding — every unique category becomes "
            "its own 0/1 column (for example, a 'Gender' column with 'Male'/'Female' becomes two columns: 'Gender_Male' and "
            "'Gender_Female', each containing only 0 or 1)."
        )

    if not feature_numeric and not feature_categorical:
        st.info("No feature columns were detected.")

    transformers = []
    if feature_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, feature_numeric))

    if feature_categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", categorical_pipeline, feature_categorical))

    preprocessor = ColumnTransformer(transformers)

    st.success("Preprocessing pipeline built successfully. Ready for model training.")

    return preprocessor, feature_cols