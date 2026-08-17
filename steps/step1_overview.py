import streamlit as st

def show_overview(df):
    """Step: Display dataset size, missing values, duplicates, and a preview table."""
    st.header("Step 2: Dataset Overview")
    st.write("A quick snapshot of the dataset's size and health.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Dataset Preview")
    st.write(f"Showing the first 20 of {df.shape[0]} total rows.")
    st.dataframe(df.head(20), use_container_width=True)