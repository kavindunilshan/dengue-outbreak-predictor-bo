import lightgbm as lgb
import numpy as np
import pandas as pd
import json

# Load city mapping from JSON file
with open('data/city_mapping.json', 'r') as f:
    city_mapping = json.load(f)

# Load test data (ensure 'week' is treated as a string)
df = pd.read_csv('data/xtest_ytest.csv', dtype={'week': str})

# Load the trained LightGBM model
model = lgb.Booster(model_file='data/lightgbm_dengue_model.txt')


def get_predictions_for_week(week: str):
    """Fetch outbreak predictions for a given week using the trained LightGBM classification model."""
    filtered_df = df[df['week'] == week]

    if filtered_df.empty:
        return {"error": "No data found for the given week"}

    # Convert 'week' to string
    filtered_df['week'] = filtered_df['week'].astype(int)

    # drop outbreak column
    x_test_filtered = filtered_df.drop('outbreak', axis=1)

    # Get feature names from the trained model
    model_features = model.feature_name()

    # Get feature names from the test data
    test_features = list(x_test_filtered.columns)

    # Find missing and extra columns
    missing_features = set(model_features) - set(test_features)
    extra_features = set(test_features) - set(model_features)

    print("Missing features:", missing_features)
    print("Extra features:", extra_features)

    probs = model.predict(x_test_filtered, num_iteration=model.best_iteration)  # Probabilities for each class
    y_pred = np.argmax(probs, axis=1)

    # Replace encoded city values with geocodes
    filtered_df['geocode'] = filtered_df['city_encoded'].astype(str).map(city_mapping)

    # Add predictions to the dataframe
    filtered_df['outbreak_prediction'] = y_pred

    # Convert to JSON format
    return filtered_df[['geocode', 'outbreak_prediction']].to_dict(orient='records')
