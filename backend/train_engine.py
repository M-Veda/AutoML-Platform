import json
import os

import pandas as pd
import joblib
import shap
import time

from backend.classification_models import get_classification_models
from backend.regression_models import get_regression_models

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# TRAIN FUNCTION
# =========================

def train_models(dataset_path, target_column):

    # Load dataset
    df = pd.read_csv(dataset_path)
    # =========================
    # DATASET ANALYSIS
    # =========================

    dataset_info = {

    "rows": int(df.shape[0]),

    "columns": int(df.shape[1]),

    "missing_values": int(df.isnull().sum().sum()),

    "numerical_columns": int(
        len(
            df.select_dtypes(
                include=["int64", "float64"]
            ).columns
        )
    ),

    "categorical_columns": int(
        len(
            df.select_dtypes(
                include=["object", "string"]
            ).columns
        )
    ),

    "memory_usage_mb": round(

        df.memory_usage(deep=True).sum() / 1024 / 1024,

        2
    )
}

    # Features & target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Store feature names
    feature_columns = list(X.columns)

    # Detect problem type
    if y.dtype == "object" or y.nunique() <= 20:

        problem_type = "Classification"

    else:

        problem_type = "Regression"

    # Column detection
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns

    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

    # ==========================================
    # CLASSIFICATION
    # ==========================================

    if problem_type == "Classification":

        models = get_classification_models()

        results = {}
        trained_pipelines = {}
        
        for model_name, model in models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = round(
                time.time() - start_time,
                4
            )
            trained_pipelines[model_name] = pipeline
            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted"
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted"
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted"
            )

            results[model_name] = {

                "Accuracy": round(accuracy, 4),

                "Precision": round(precision, 4),

                "Recall": round(recall, 4),

                "F1 Score": round(f1, 4),

                "Training Time": training_time
            }

        best_model = max(
            results,
            key=lambda x: results[x]["F1 Score"]
        )

        best_model_object = trained_pipelines[best_model]

    # ==========================================
    # REGRESSION
    # ==========================================

    else:

        models = get_regression_models()

        results = {}
        trained_pipelines = {}

        for model_name, model in models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = round(
                time.time() - start_time,
                4
            )
            trained_pipelines[model_name] = pipeline
            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)

            rmse = mse ** 0.5

            mae = mean_absolute_error(y_test, y_pred)

            r2 = r2_score(y_test, y_pred)

            results[model_name] = {

                "MSE": round(mse, 4),

                "RMSE": round(rmse, 4),

                "MAE": round(mae, 4),

                "R2 Score": round(r2, 4),

                "Training Time": training_time
            }

        best_model = min(
            results,
            key=lambda x: results[x]["RMSE"]
        )

        best_model_object = trained_pipelines[best_model]


    # =========================
    # FEATURE IMPORTANCE
    # =========================

    feature_importance = {}

    try:

        rf_pipeline = trained_pipelines.get("Random Forest")

        if rf_pipeline:

            rf_model = rf_pipeline.named_steps["model"]

            importance_values = rf_model.feature_importances_

            for feature, importance in zip(
                feature_columns,
                importance_values
            ):

                feature_importance[feature] = float(importance)

    except:

        feature_importance = {}

    # =========================
    # SHAP EXPLAINABILITY
    # =========================

    shap_values_output = {}

    try:

        explainer = shap.Explainer(
            best_model_object,
            X_train
        )

        shap_values = explainer(X_test[:1])

        for feature, value in zip(
            feature_columns,
            shap_values.values[0]
        ):

            shap_values_output[feature] = float(value)

    except:

        shap_values_output = {}

    # =========================
    # SAVE FILES
    # =========================

    best_model_object = trained_pipelines[best_model]

    joblib.dump(
        best_model_object,
        "outputs/best_model.pkl"
    )

    # Save feature columns
    joblib.dump(
        list(X.columns),
        "outputs/feature_columns.pkl"
    )

    # Save problem type
    joblib.dump(
        problem_type,
        "outputs/problem_type.pkl"
    )

    # =========================
    # RETURN RESULTS
    # =========================

    final_results = {

    "problem_type": problem_type,

    "dataset_info": dataset_info,

    "best_model": best_model,

    "results": results,

    "feature_columns": feature_columns,

    "dataset_preview": df.head(5).to_dict(orient="records"),

    "feature_importance": feature_importance,

    "shap_values": shap_values_output
}

    import json

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/training_results.json", "w") as f:

        json.dump(
            final_results,
            f,
            indent=4
        )

    return final_results