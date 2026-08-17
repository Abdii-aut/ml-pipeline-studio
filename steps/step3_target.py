import streamlit as st

def select_target_and_detect(df):
    """Step: Let user pick target column, auto-detect regression vs classification."""
    st.header("Step 4: Select Target & Detect Problem Type")
    st.write("Choose the column you want to predict. The platform will automatically determine whether this is a regression or classification problem.")

    target_col = st.selectbox("Select your target column", df.columns)

    unique_values = df[target_col].nunique()
    dtype = df[target_col].dtype
    total_rows = len(df)
    uniqueness_ratio = unique_values / total_rows if total_rows > 0 else 0

    if dtype == "object" or unique_values <= 10:
        problem_type = "Classification"
        explanation = f"'{target_col}' has only {unique_values} unique value(s), so this is treated as a classification problem (predicting a category)."
    else:
        problem_type = "Regression"
        explanation = f"'{target_col}' has {unique_values} unique numeric values, so this is treated as a regression problem (predicting a continuous number)."

    st.info(f"**Detected Problem Type: {problem_type}**\n\n{explanation}")

    if uniqueness_ratio > 0.9:
        st.warning(
            f"'{target_col}' has {unique_values} unique values out of {total_rows} rows — nearly every value is different. "
            "This often means the column is an ID or identifier rather than something meaningful to predict. "
            "Consider selecting a different target column."
        )

    return target_col, problem_type