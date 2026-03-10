
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_PATH, MODEL_NAME, ALIAS 
import pickle
import os
import mlflow

def save_model(model: BaseEstimator, name: str):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    
    if not os.path.exists(MODEL_PATH) : 
        os.mkdir(MODEL_PATH)
    with open(os.path.join(MODEL_PATH, name),"wb") as f:
        pickle.dump(model, f)
        

def load_model(name: str) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(os.path.join(MODEL_PATH, name),"rb")  as f:
        model = pickle.load(f)
    return model

def load_model_mkflow() -> BaseEstimator:
    """Load the model from the specified path using mkflow."""
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return model