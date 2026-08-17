import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_feature_importance(results, feature_cols, numeric_cols, categorical_cols):
    """Step: Show which features the best model relied on most, if supported."""
    best_pipeline = results.get("best_pipeline")
    best_model_name = results.get("best_model_name")

    if best_pipeline is None:
        return

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return

    st.header("Step 7.2: Feature Importance")
    st.write(f"Which columns did {best_model_name} rely on most when making predictions?")

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importances = model.feature_importances_

    clean_names = [name.split("__")[-1] for name in feature_names]

    importance_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    top_n = min(15, len(importance_df))
    top_df = importance_df.head(top_n)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.3)))
        sns.barplot(data=top_df, x="Importance", y="Feature", ax=ax, color="#3B82F6")
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.caption(
        "How to read this: Each bar shows how much a feature contributed to the model's decisions overall — "
        "longer bars mean the model relied on that feature more heavily when making predictions. Note that after "
        "One-Hot Encoding, a single original column (like 'Gender') may appear as several separate bars "
        "(e.g., 'Gender_Male', 'Gender_Female') since each category is treated as its own input."
    )

    with st.expander("Full Feature Importance Table"):
        st.dataframe(importance_df, use_container_width=True)