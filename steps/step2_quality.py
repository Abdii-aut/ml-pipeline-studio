import streamlit as st
import pandas as pd

def show_data_quality(df, numeric_cols, categorical_cols):
    """Step: Report missing values, duplicates, and column type breakdown."""
    st.header("Step 3: Data Quality Report")
    st.write("This report automatically checks every column for common real-world data problems.")

    colA, colB = st.columns(2)
    with colA:
        st.write(f"**🔢 Numeric Columns ({len(numeric_cols)}):**")
        st.write(numeric_cols if numeric_cols else "None found")
    with colB:
        st.write(f"**🔤 Categorical Columns ({len(categorical_cols)}):**")
        st.write(categorical_cols if categorical_cols else "None found")

    st.subheader("Missing Value Check")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    missing_report = missing_report[missing_report["Missing Count"] > 0]

    if missing_report.empty:
        st.success("✅ No missing values detected — this dataset is clean.")
    else:
        st.warning(f"⚠️ {len(missing_report)} column(s) have missing values. These will be automatically handled during preprocessing (numeric → median, categorical → most frequent).")
        st.dataframe(missing_report, use_container_width=True)

    st.subheader("Duplicate Row Check")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠️ {dup_count} duplicate row(s) found. Consider reviewing these before training.")