import joblib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from preprocessing import manual_preprocessing



def train():
    """Trains a linear regression model on the full dataset and stores output."""

    # Load the data
    data = pd.read_csv("data/model_input.csv")

    # Define features to use
    num_features = ['nbr_bedrooms', 'primary_energy_consumption_sqm', 
                    'total_area_sqm', 'garden_sqm', 'terrace_sqm', 'surface_land_sqm']
    fl_features = ['fl_furnished', 'fl_terrace', 'fl_garden', 'fl_swimming_pool']
    cat_features = ['property_type', 'region', 'state_building']

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')
    
    # Preprocessing for categorical data
    cat_imputer = SimpleImputer(strategy='most_frequent')
    enc = OneHotEncoder(handle_unknown='ignore')
    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', cat_imputer),
        ('enc', enc)
        ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical_transformer', numerical_transformer, num_features + fl_features),
            ('categorical_transformer', categorical_transformer, cat_features)
        ])

    # Define the scaler
    scaler = StandardScaler()

    # Define the model
    model = LinearRegression()

    # Create and evaluate the pipeline (preprocessing and modeling code)
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('scaler', scaler),
                                  ('model', model)
                                  ])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)

    print(f"Features: \n {X_train.columns.tolist()}")

    # Evaluate the model
    train_score = r2_score(y_train, my_pipeline.predict(X_train))
    test_score = r2_score(y_test, my_pipeline.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "pipeline": {
            "preprocessor": preprocessor, 
            "scaler": scaler,
            "model": model,
        }
    }

    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    manual_preprocessing("data/properties.csv", "data/model_input.csv")
    train()
