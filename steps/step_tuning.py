import streamlit as st
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

PARAM_GRIDS = {
    "Decision Tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5],
    },
    "LightGBM": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
    },
    "CatBoost": {
        "model__iterations": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    },
    "Logistic Regression": {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
    "Linear Regression": {},
    "SVM": {
        "model__C": [0.1, 1, 10],
    },
    "Naive Bayes": {},
}


def show_tuning_section(results, X, y, problem_type):
    """Step: Let the user fine-tune the best model's hyperparameters using RandomizedSearchCV,
    compared fairly against the same model's default cross-validated score."""
    best_model_name = results.get("best_model_name")
    best_pipeline = results.get("best_pipeline")

    if best_pipeline is None:
        return

    st.header("Step 7.3: Hyperparameter Tuning")
    st.write(
        f"The best model, {best_model_name}, was trained using default settings. "
        "Hyperparameter tuning searches through several configurations to find settings that may improve performance further."
    )

    param_grid = PARAM_GRIDS.get(best_model_name, {})

    if not param_grid:
        st.info(f"{best_model_name} has no tunable parameters configured for this platform, or performs best with default settings.")
        return

    st.write("Parameter combinations that will be sampled from:")
    st.json(param_grid)

    st.caption(
        "To keep this fast and avoid overloading your machine, tuning uses a randomized search "
        "(a limited number of random combinations, evaluated with 3-fold cross-validation) instead of "
        "testing every possible combination, and runs on a single CPU core. The baseline score below is "
        "measured using the same 3-fold cross-validation, so the comparison is fair — both scores use "
        "identical methodology."
    )

    if st.button("Run Hyperparameter Tuning", type="primary"):
        scoring = "r2" if problem_type == "Regression" else "f1_weighted"

        with st.spinner(f"Measuring {best_model_name}'s baseline cross-validated performance..."):
            baseline_scores = cross_val_score(best_pipeline, X, y, cv=3, scoring=scoring)
            baseline_score = baseline_scores.mean()

        with st.spinner(f"Testing parameter combinations for {best_model_name}. This may take a moment..."):
            search = RandomizedSearchCV(
                best_pipeline,
                param_grid,
                n_iter=8,
                cv=3,
                scoring=scoring,
                n_jobs=1,
                random_state=42,
            )
            search.fit(X, y)

        st.success("Tuning complete.")

        tuned_score = search.best_score_
        improvement = tuned_score - baseline_score

        col1, col2 = st.columns(2)
        col1.metric("Default Settings (3-fold CV)", f"{baseline_score:.4f}")
        col2.metric("Tuned Settings (3-fold CV)", f"{tuned_score:.4f}", delta=f"{improvement:.4f}")

        if improvement > 0.001:
            st.success(f"Tuning improved performance by {improvement:.4f} ({improvement / abs(baseline_score) * 100:.1f}% relative improvement) over default settings.")
        elif improvement < -0.001:
            st.info("The default settings were already close to optimal for this dataset — the search did not find a meaningfully better configuration.")
        else:
            st.info("Tuning found settings roughly equivalent to the defaults — the model was already well-configured.")

        st.write("Best parameters found:")
        st.json(search.best_params_)

        st.caption(
            "Both scores above use identical 3-fold cross-validation, so this is a fair, apples-to-apples comparison "
            "of default vs. tuned hyperparameters."
        )

        return search.best_estimator_

    return None