import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_visualizations(df, numeric_cols, categorical_cols, target_col):
    """Step: Visualize the dataset using matplotlib and seaborn before training."""
    st.header("Step 6: Data Visualization")
    st.write("Before training, it helps to visually understand the data — distributions, relationships, and outliers.")

    MAX_PLOT_ROWS = 5000
    if len(df) > MAX_PLOT_ROWS:
        st.info(
            f"This dataset has {len(df):,} rows. To keep visualizations fast, plots below are generated "
            f"from a random sample of {MAX_PLOT_ROWS:,} rows. This does not affect model training, which "
            "still uses the full dataset."
        )
        plot_df = df.sample(n=MAX_PLOT_ROWS, random_state=42)
    else:
        plot_df = df
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.size": 8})

    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                        annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
            ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        st.caption(
            "What this shows: How strongly every pair of numeric columns relates to each other, from -1 to +1. "
            "+1 (dark red) means the columns increase together. -1 (dark blue) means one increases as the other decreases. "
            "Close to 0 means little to no relationship. If two features are very highly correlated (0.9+), they carry mostly "
            "the same information and may make some models less stable."
        )

    st.subheader(f"Target Column Distribution: '{target_col}'")

    target_unique_count = df[target_col].nunique()
    MAX_CATEGORIES_TO_PLOT = 20

    if target_col not in numeric_cols and target_unique_count > MAX_CATEGORIES_TO_PLOT:
        st.warning(
            f"'{target_col}' has {target_unique_count} unique values, which is too many to plot as a "
            f"bar chart. Showing the {MAX_CATEGORIES_TO_PLOT} most common values instead."
        )
        top_counts = df[target_col].value_counts().head(MAX_CATEGORIES_TO_PLOT)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=top_counts.values, y=top_counts.index.astype(str), ax=ax, color="steelblue")
            ax.set_xlabel("Count", fontsize=8)
            ax.set_ylabel(target_col, fontsize=8)
            ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            if target_col in numeric_cols:
                sns.histplot(plot_df[target_col], kde=True, ax=ax, color="steelblue")
            else:
                sns.countplot(x=plot_df[target_col], ax=ax, palette="viridis")
            ax.set_xlabel(target_col, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
    if target_col in numeric_cols:
        st.caption(
            "What this shows: Which range of target values occurs most often. A symmetric, bell-shaped curve "
            "means the data is roughly normally distributed, which tends to work well with models like Linear Regression. "
            "A long tail on one side (skew) means a few extreme values exist, which can pull predictions off for those cases."
        )
    else:
        st.caption(
            "What this shows: How many rows belong to each category. If one bar is much taller than the others, the "
            "dataset is imbalanced — the model can become biased toward predicting the majority class, which is why "
            "F1 Score matters more than plain Accuracy in this situation."
        )

    feature_numeric = [c for c in numeric_cols if c != target_col]
    if feature_numeric:
        st.subheader("Numeric Feature Distributions")
        selected_col = st.selectbox("Select a numeric feature to visualize", feature_numeric)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df[selected_col], kde=True, ax=ax, color="mediumseagreen")
            ax.set_title(f"Distribution of {selected_col}", fontsize=9)
            ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(x=df[selected_col], ax=ax, color="salmon")
            ax.set_title(f"Boxplot of {selected_col}", fontsize=9)
            ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        st.caption(
            "Histogram (left): Shows the spread of values for this feature — where most data points cluster, and how wide the range is. "
            "Boxplot (right): Summarizes the same feature into quartiles. Dots outside the box are outliers — unusual "
            "values far from the rest of the data. Outliers can distort how a model learns, especially for models sensitive "
            "to scale like Linear Regression or KNN."
        )

    feature_categorical = [c for c in categorical_cols if c != target_col]
    if feature_categorical:
        st.subheader("Categorical Feature Counts")
        selected_cat = st.selectbox("Select a categorical feature to visualize", feature_categorical)

        cat_unique_count = df[selected_cat].nunique()

        if cat_unique_count > MAX_CATEGORIES_TO_PLOT:
            st.warning(
                f"'{selected_cat}' has {cat_unique_count} unique values, which is too many to plot as a "
                f"bar chart. Showing the {MAX_CATEGORIES_TO_PLOT} most common values instead."
            )
            top_counts = df[selected_cat].value_counts().head(MAX_CATEGORIES_TO_PLOT)
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.barplot(x=top_counts.values, y=top_counts.index.astype(str), ax=ax, color="lightpink")
                ax.set_xlabel("Count", fontsize=8)
                ax.set_ylabel(selected_cat, fontsize=8)
                ax.tick_params(labelsize=7)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.countplot(x=plot_df[selected_cat], ax=ax, palette="pastel")
                ax.set_title(f"Count of {selected_cat}", fontsize=9)
                ax.tick_params(labelsize=7, rotation=30)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

        st.caption(
            "What this shows: How often each category appears in the dataset. Tall bars are well-represented categories; "
            "short bars are rare ones. A category that appears only a handful of times contributes very little signal to the "
            "model after encoding, especially in smaller datasets."
        )