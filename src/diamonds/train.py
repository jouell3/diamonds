import pandas as pd
import os
import loguru
from diamonds.data import load_data, clean_data, create_X_y, split_data
from diamonds.model import create_model, create_preproc, train_model, evaluate_model, predict, preprocess_data
from diamonds.registry import save_model, load_model
import mlflow

logger = loguru.logger

def train(
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end‑to‑end pipeline:

    - load and clean the raw data
    - preprocess it and build X, y
    - split into train / test
    - build the model and preprocessing
    - train, evaluate, and save the trained model
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("diamonds")

    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    preproc = create_preproc(X_train)
    X_train_scaled = preprocess_data(preproc, X_train)
    X_test_scaled = preprocess_data(preproc, X_test)
    model_ = create_model(model_name)
    model_ = train_model(model_, X_train_scaled, y_train)

    
if __name__ == "__main__":
    train()

