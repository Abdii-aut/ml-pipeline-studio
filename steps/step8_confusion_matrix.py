import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def show_confusion_matrix(results, problem_type):
    """Step: Show a confusion matrix for classification problems, with an explanation."""
    if problem_type != "Classification":
        return

    best_pipeline = results.get("best_pipeline")
    best_model_name = results.get("best_model_name")
    X_test = results.get("X_test")
    y_test = results.get("y_test")
    label_encoder = results.get("label_encoder")

    if best_pipeline is None or X_test is None:
        return

    st.header("Step 7.1: Confusion Matrix")
    st.write(f"A closer look at exactly what kinds of mistakes **{best_model_name}** makes on the test data.")

    preds = best_pipeline.predict(X_test)

    labels = sorted(y_test.unique())
    display_labels = labels
    if label_encoder is not None:
        display_labels = label_encoder.inverse_transform(labels)

    cm = confusion_matrix(y_test, preds, labels=labels)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=display_labels, yticklabels=display_labels,
            annot_kws={"size": 9}
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.caption(
        "**How to read this:** Each row is the *actual* class, each column is what the model *predicted*. "
        "The diagonal (top-left to bottom-right) shows correct predictions — the higher these numbers, the better. "
        "Everything off the diagonal is a mistake: for example, a value in row 'Yes' and column 'No' means the model "
        "predicted 'No' when the real answer was 'Yes' — this is exactly the kind of error Recall measures."
    )

    with st.expander("📋 Detailed Classification Report (per class)"):
        report = classification_report(
            y_test, preds, target_names=[str(l) for l in display_labels], zero_division=0
        )
        st.code(report)
        st.caption(
            "This breaks down Precision, Recall, and F1 Score **per class**, rather than as one overall number. "
            "This is especially useful when one class is much rarer than another — the overall F1 Score can look "
            "fine even if the model performs poorly specifically on the rare class."
        )