
from sklearn.base import BaseEstimator
from diamonds import model
from diamonds.params import MODEL_PATH, MODEL_NAME, ALIAS, MODEL_REGISTRY, MODEL_MLFLOW_URI, PREPROD_MODEL_NAME, MODEL_NAME 
import pickle
import os
import mlflow

mlflow.set_tracking_uri(MODEL_MLFLOW_URI)

def save_model(model: BaseEstimator, 
               name: str, 
               MODEL_REGISTRY: str = MODEL_REGISTRY) -> None:
    
    """Save the model to the specified path."""
        
        # Implement the logic to save the model (e.g., using pickle, joblib, etc.)

    if MODEL_REGISTRY == "mlflow" and name == PREPROD_MODEL_NAME : 
        mlflow.sklearn.log_model(f"models:/{PREPROD_MODEL_NAME}@{ALIAS}")
    elif MODEL_REGISTRY == "mlflow" and name == MODEL_NAME : 
        mlflow.sklearn.log_model(f"models:/{MODEL_NAME}@{ALIAS}")
    elif MODEL_REGISTRY == "local" and name == PREPROD_MODEL_NAME : 
        if not os.path.exists(MODEL_PATH) : 
            os.mkdir(MODEL_PATH)
        with open(os.path.join(MODEL_PATH, f'{PREPROD_MODEL_NAME}.pkl'),"wb") as f:
            pickle.dump(model, f)
    elif MODEL_REGISTRY == "local" and name == MODEL_NAME :
        if not os.path.exists(MODEL_PATH) : 
            os.mkdir(MODEL_PATH)
        with open(os.path.join(MODEL_PATH, f'{MODEL_NAME}.pkl'),"wb") as f:
            pickle.dump(model, f)
        

def load_model(name: str, 
               MODEL_REGISTRY: str = MODEL_REGISTRY) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g.os.nameing pickle, joblib, etc.)
    estimator_path = os.path.join(MODEL_PATH, f"{name}.pkl")
    if MODEL_REGISTRY == "mlflow" and name == PREPROD_MODEL_NAME : 
        model = mlflow.sklearn.load_model(f"models:/{PREPROD_MODEL_NAME}@{ALIAS}")
    elif MODEL_REGISTRY == "mlflow" and name == MODEL_NAME : 
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    elif MODEL_REGISTRY == "local" and name == PREPROD_MODEL_NAME : 
        with open(os.path.join(MODEL_PATH, f"{PREPROD_MODEL_NAME}.pkl"),"rb")  as f:
            model = pickle.load(f)
    elif MODEL_REGISTRY == "local" and name == MODEL_NAME : 
        with open(os.path.join(MODEL_PATH, f"{MODEL_NAME}.pkl"),"rb")  as f:
            model = pickle.load(f)
    return model

def load_model_mkflow() -> BaseEstimator:
    """Load the model from the specified path using mkflow."""
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return model