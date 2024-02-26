import joblib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from preprocessing import manual_preprocessing



def train():
    """Trains a model on the full dataset and save the model."""

    # Load the data
    data = pd.read_csv("data/model_input.csv")

    # Define features to use
    num_features = ['nbr_bedrooms', 'primary_energy_consumption_sqm', 
                    'total_area_sqm', 'garden_sqm', 'terrace_sqm', 'surface_land_sqm']
    fl_features = ['fl_furnished', 'fl_terrace', 'fl_garden', 'fl_swimming_pool']
    cat_features = ['property_type', 'region', 'state_building']

    # Shuffle the data
    data = data.sample(frac=1, random_state=505).reset_index(drop=True)

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))
        ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical_transformer', numerical_transformer, num_features + fl_features),
            ('categorical_transformer', categorical_transformer, cat_features)
        ])

    # Define the model
    # model = LinearRegression()
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='absolute_error')
    model = RandomForestRegressor(n_estimators=100, random_state=505)

    # Create and evaluate the pipeline (preprocessing and modeling code)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)
                                ])

    # Preprocessing of training data, fit model 
    pipeline.fit(X_train, y_train)

    print(f"Features: \n {X_train.columns.tolist()}")

    # Evaluate the model
    train_score = r2_score(y_train, pipeline.predict(X_train))
    test_score = r2_score(y_test, pipeline.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "pipeline": pipeline
    }

    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    manual_preprocessing("data/properties.csv", "data/model_input.csv")
    train()
