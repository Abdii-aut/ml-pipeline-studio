import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_models(problem_type):
    """Return a dictionary of {model_name: model_object} based on problem type."""
    models = {}

    if problem_type == "Regression":
        models["Linear Regression"] = LinearRegression()
        models["Decision Tree"] = DecisionTreeRegressor(random_state=42)
        models["Random Forest"] = RandomForestRegressor(random_state=42, n_estimators=200)
        models["KNN"] = KNeighborsRegressor()
        models["SVM"] = SVR()
        models["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostRegressor(random_state=42, verbose=0)
    else:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
        models["Decision Tree"] = DecisionTreeClassifier(random_state=42)
        models["Random Forest"] = RandomForestClassifier(random_state=42, n_estimators=200)
        models["KNN"] = KNeighborsClassifier()
        models["SVM"] = SVC(probability=True, random_state=42)
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
        models["Naive Bayes"] = GaussianNB()
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss")
        if HAS_LGBM:
            models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostClassifier(random_state=42, verbose=0)

    return models

def train_and_compare_models(df, target_col, problem_type, preprocessor, feature_cols):
    """Step: Train every model, evaluate on the same test set (plus cross-validation), and show comparison + best model."""
    st.header("Step 7: Train & Compare Models")
    st.write("The platform now trains every applicable algorithm on the same training data, and evaluates all of them on the same unseen test data — this ensures a fair comparison.")

    valid_rows = df[target_col].notna()
    dropped_count = (~valid_rows).sum()

    if dropped_count > 0:
        st.warning(f"{dropped_count} row(s) had a missing target value and were excluded from training (a model can't learn from an unknown answer).")

    X = df.loc[valid_rows, feature_cols]
    y = df.loc[valid_rows, target_col].copy()

    LARGE_DATASET_THRESHOLD = 20000

    if len(X) > LARGE_DATASET_THRESHOLD:
        st.info(
            f"This dataset has {len(X):,} rows. Training all 9 models with cross-validation on the full "
            "dataset can take a long time, especially for SVM and KNN. The options below can speed this up."
        )

        col1, col2 = st.columns(2)
        with col1:
            use_sample = st.checkbox("Train on a random sample instead of the full dataset", value=True)
        with col2:
            skip_slow_models = st.checkbox("Skip SVM and KNN (slowest on large datasets)", value=True)

        if use_sample:
            sample_size = st.slider("Sample size for training", min_value=2000, max_value=min(len(X), 50000), value=min(10000, len(X)), step=1000)
            sample_idx = X.sample(n=sample_size, random_state=42).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]
            st.caption(f"Training on a random sample of {sample_size:,} rows out of {len(valid_rows):,} total.")

        cv_folds = 3
    else:
        skip_slow_models = False
        cv_folds = 5
    empty_result = {
        "best_model_name": None,
        "best_pipeline": None,
        "label_encoder": None,
        "results_df": pd.DataFrame(),
        "primary_metric": None,
        "X_test": None,
        "y_test": None,
    }

    if len(X) < 10:
        st.error(
            f"Only {len(X)} row(s) remain after removing missing target values — this is too few to train any model reliably. "
            "Please select a different target column, or upload a dataset with more complete data for this column."
        )
        return empty_result

    label_encoder = None
    if problem_type == "Classification" and not pd.api.types.is_integer_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    can_stratify = False
    if problem_type == "Classification" and y.nunique() > 1:
        class_counts = y.value_counts()
        can_stratify = class_counts.min() >= 2

    stratify = y if can_stratify else None

    if problem_type == "Classification" and not can_stratify:
        st.warning("Some categories in the target column appear only once, so a perfectly balanced train/test split isn't possible. Proceeding with a regular random split instead.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError as e:
        st.error(f"Unable to split the data into training and test sets: {e}")
        empty_result["label_encoder"] = label_encoder
        return empty_result

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Not enough data remains to create both a training set and a test set. Please select a different target column or upload more data.")
        empty_result["label_encoder"] = label_encoder
        return empty_result

    st.write(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    models = get_models(problem_type)

    if skip_slow_models:
        for slow_model in ["SVM", "KNN"]:
            models.pop(slow_model, None)

    results = []
    fitted_pipelines = {}
    primary_metric = "R2 Score" if problem_type == "Regression" else "F1 Score"
    cv_scoring = "r2" if problem_type == "Regression" else "f1_weighted"

    st.write(
        f"In addition to the single train/test split above, each model is also evaluated using "
        f"{cv_folds}-fold cross-validation — the training data is split {cv_folds} different ways, the model is "
        f"trained and tested {cv_folds} times, and the results are averaged. This gives a more reliable picture "
        "of performance than relying on a single split alone."
    )

    progress_bar = st.progress(0, text="Starting training...")
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        progress_bar.progress(i / total, text=f"Training {name}...")
        try:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            if problem_type == "Regression":
                metrics = {
                    "R2 Score": r2_score(y_test, preds),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
                    "MAE": mean_absolute_error(y_test, preds),
                }
            else:
                avg = "binary" if y.nunique() == 2 else "weighted"
                metrics = {
                    "Accuracy": accuracy_score(y_test, preds),
                    "F1 Score": f1_score(y_test, preds, average=avg, zero_division=0),
                    "Precision": precision_score(y_test, preds, average=avg, zero_division=0),
                    "Recall": recall_score(y_test, preds, average=avg, zero_division=0),
                }

            try:
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=cv_scoring)
                metrics["CV Mean"] = cv_scores.mean()
                metrics["CV Std"] = cv_scores.std()
            except Exception:
                metrics["CV Mean"] = None
                metrics["CV Std"] = None

            results.append({"Model": name, **metrics})
            fitted_pipelines[name] = pipe

        except Exception as e:
            results.append({"Model": name, "Status": f"Failed: {e}"})

    progress_bar.progress(1.0, text="Training complete.")

    results_df = pd.DataFrame(results)

    if primary_metric in results_df.columns:
        ranked_df = results_df.dropna(subset=[primary_metric]).sort_values(
            by=primary_metric, ascending=False
        ).reset_index(drop=True)
    else:
        ranked_df = results_df

    st.subheader("Model Comparison")
    st.write(
        f"Models are ranked by {primary_metric} (from the single test split) — the standard metric for "
        f"{problem_type.lower()} problems. CV Mean and CV Std show the average score and consistency "
        "across 5 cross-validation folds — a model with a high CV Mean and low CV Std is both accurate and stable."
    )
    st.dataframe(ranked_df, use_container_width=True)

    if len(ranked_df) == 0:
        st.error("All models failed to train. Please check the dataset for issues (see the Status column above).")
        return {
            "best_model_name": None,
            "best_pipeline": None,
            "label_encoder": label_encoder,
            "results_df": results_df,
            "primary_metric": primary_metric,
            "X_test": X_test,
            "y_test": y_test,
        }

    best_model_name = ranked_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines.get(best_model_name)
    best_score = ranked_df.iloc[0][primary_metric]
    best_cv_mean = ranked_df.iloc[0].get("CV Mean")

    st.success(f"Best Model: {best_model_name} ({primary_metric} = {best_score:.4f})")
    if pd.notna(best_cv_mean):
        st.write(f"This model also achieved a 5-fold cross-validation average of {best_cv_mean:.4f}, confirming its performance holds up across different splits of the data — not just the one test set used above.")
    st.write("This model outperformed all others on unseen test data and will be used for predictions.")

    return {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "results_df": ranked_df,
        "primary_metric": primary_metric,
        "X_test": X_test,
        "y_test": y_test,
    }