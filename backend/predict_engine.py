import joblib
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "outputs/best_model.pkl"
)

feature_columns = joblib.load(
    "outputs/feature_columns.pkl"
)

# =========================
# PREDICTION FUNCTION
# =========================

def make_prediction(features):

    cleaned_features = {}

    for key, value in features.items():

        try:

            cleaned_features[key] = float(value)

        except:

            cleaned_features[key] = value

    # Create dataframe
    input_df = pd.DataFrame(
        [cleaned_features]
    )

    # Reorder columns
    input_df = input_df[
        feature_columns
    ]

    # Prediction
    prediction = model.predict(
        input_df
    )[0]

    # Convert numpy types
    if isinstance(prediction, np.generic):

        prediction = prediction.item()

    return {

        "prediction": prediction
    }