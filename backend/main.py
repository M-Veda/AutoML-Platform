import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# CLASSIFICATION MODELS
# =========================

from classification_models import get_classification_models
from regression_models import get_regression_models

# =========================
# EVALUATION METRICS
# =========================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("datasets/sample.csv")

# =========================
# DATASET ANALYSIS
# =========================

print("\n================ DATASET PREVIEW ================\n")
print(df.head())

print("\n================ DATASET SHAPE ================\n")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n================ COLUMN NAMES ================\n")
print(df.columns.tolist())

print("\n================ DATA TYPES ================\n")
print(df.dtypes)

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

print("\n================ DUPLICATE ROWS ================\n")
print("Duplicate Rows:", df.duplicated().sum())

# =========================
# TARGET COLUMN
# =========================

target_column = input("\nEnter Target Column Name: ")

# =========================
# PROBLEM TYPE DETECTION
# =========================

if df[target_column].dtype == "object":
    problem_type = "Classification"

elif df[target_column].nunique() < 20:
    problem_type = "Classification"

else:
    problem_type = "Regression"

print("\n================ PROBLEM TYPE ================\n")
print("Detected Problem Type:", problem_type)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop(columns=[target_column])
y = df[target_column]

# =========================
# COLUMN DETECTION
# =========================

numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns

categorical_columns = X.select_dtypes(include=["object", "string"]).columns

print("\n================ NUMERICAL COLUMNS ================\n")
print(numerical_columns)

print("\n================ CATEGORICAL COLUMNS ================\n")
print(categorical_columns)

# =========================
# NUMERICAL PIPELINE
# =========================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# =========================
# CATEGORICAL PIPELINE
# =========================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# =========================
# COMBINE PIPELINES
# =========================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# =========================
# APPLY PREPROCESSING
# =========================

X_processed = preprocessor.fit_transform(X)

print("\n================ PREPROCESSING COMPLETED ================\n")

print("Processed Feature Shape:", X_processed.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)

print("\n================ TRAIN TEST SPLIT ================\n")

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

# =========================
# CLASSIFICATION MODELS
# =========================

if problem_type == "Classification":

    models = get_classification_models()

    results = {}

    print("\n================ TRAINING MODELS ================\n")

    for model_name, model in models.items():

        print(f"Training {model_name}...")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

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

        # Store results
        results[model_name] = {

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1
        }

    # =========================
    # DISPLAY RESULTS
    # =========================

    print("\n================ MODEL RESULTS ================\n")

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        for metric_name, value in metrics.items():

            print(f"{metric_name}: {value:.4f}")

    # =========================
    # BEST MODEL
    # =========================

    best_model = max(
        results,
        key=lambda x: results[x]["F1 Score"]
    )

    print("\n================ BEST MODEL ================\n")

    print("Best Model:", best_model)

    print(
        "Best F1 Score:",
        round(results[best_model]["F1 Score"], 4)
    )

    # =========================
    # SAVE BEST MODEL
    # =========================

    best_model_object = models[best_model]

    joblib.dump(
        best_model_object,
        "outputs/best_model.pkl"
    )

    print("\nBest model saved successfully!")

# =========================
# REGRESSION MODELS
# =========================

elif problem_type == "Regression":

    models = get_regression_models()

    results = {}

    print("\n================ TRAINING MODELS ================\n")

    for model_name, model in models.items():

        print(f"Training {model_name}...")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)

        rmse = mse ** 0.5

        mae = mean_absolute_error(y_test, y_pred)

        r2 = r2_score(y_test, y_pred)

        # Store results
        results[model_name] = {

            "MSE": mse,

            "RMSE": rmse,

            "MAE": mae,

            "R2 Score": r2
        }

    # =========================
    # DISPLAY RESULTS
    # =========================

    print("\n================ MODEL RESULTS ================\n")

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        for metric_name, value in metrics.items():

            print(f"{metric_name}: {value:.4f}")

    # =========================
    # BEST MODEL
    # =========================

    best_model = min(
        results,
        key=lambda x: results[x]["RMSE"]
    )

    print("\n================ BEST MODEL ================\n")

    print("Best Model:", best_model)

    print(
        "Best RMSE:",
        round(results[best_model]["RMSE"], 4)
    )

    # =========================
    # SAVE BEST MODEL
    # =========================

    best_model_object = models[best_model]

    joblib.dump(
        best_model_object,
        "outputs/best_model.pkl"
    )

    print("\nBest model saved successfully!")

# =========================
# VISUALIZATION
# =========================

results_df = pd.DataFrame(results).T

print("\n================ RESULTS DATAFRAME ================\n")

print(results_df)

# ====================================================
# CLASSIFICATION VISUALIZATIONS
# ====================================================

if problem_type == "Classification":

    # Accuracy Graph

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=results_df.index,
        y=results_df["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")

    plt.xlabel("Algorithms")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig("outputs/accuracy_comparison.png")

    plt.show()

    # F1 Score Graph

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=results_df.index,
        y=results_df["F1 Score"]
    )

    plt.title("Model F1 Score Comparison")

    plt.xlabel("Algorithms")

    plt.ylabel("F1 Score")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig("outputs/f1_score_comparison.png")

    plt.show()

# ====================================================
# REGRESSION VISUALIZATIONS
# ====================================================

elif problem_type == "Regression":

    # RMSE Graph

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=results_df.index,
        y=results_df["RMSE"]
    )

    plt.title("Model RMSE Comparison")

    plt.xlabel("Algorithms")

    plt.ylabel("RMSE")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig("outputs/rmse_comparison.png")

    plt.show()

    # R2 Score Graph

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=results_df.index,
        y=results_df["R2 Score"]
    )

    plt.title("Model R2 Score Comparison")

    plt.xlabel("Algorithms")

    plt.ylabel("R2 Score")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig("outputs/r2_score_comparison.png")

    plt.show()

# =========================
# USER PREDICTION SYSTEM
# =========================

print("\n================ USER PREDICTION ================\n")

user_input = []

# Take input for every feature column

for column in X.columns:

    value = input(f"Enter value for {column}: ")

    # Convert numerical values
    if column in numerical_columns:

        value = float(value)

    user_input.append(value)

# Create dataframe

user_df = pd.DataFrame(
    [user_input],
    columns=X.columns
)

# Apply preprocessing

user_processed = preprocessor.transform(user_df)

# Load saved model

loaded_model = joblib.load(
    "outputs/best_model.pkl"
)

# Predict

prediction = loaded_model.predict(user_processed)

# =========================
# DISPLAY PREDICTION
# =========================

print("\n================ PREDICTION RESULT ================\n")

if problem_type == "Classification":

    print("Predicted Class:", prediction[0])

elif problem_type == "Regression":

    print("Predicted Value:", round(prediction[0], 2))