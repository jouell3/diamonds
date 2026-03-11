import pandas as pd
from diamonds.model import predict, preprocess_data
from diamonds.registry import load_model

import loguru

logger = loguru.logger

def predict_values(X: pd.DataFrame) -> dict[str, float]:
    """Evaluate the model on the given data and return predicted value."""
    preproc = load_model("preproc")
    prod_model = load_model("trained_model")
    logger.info("Model loaded successfully for prediction")
    X_scaled = preprocess_data(preproc, X)
    logger.info("Data preprocessed successfully for prediction")
    
    y_pred = predict(X_scaled)
    logger.info("Predictions made for prediction function")

    return y_pred
    