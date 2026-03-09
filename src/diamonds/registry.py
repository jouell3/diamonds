
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_REGISTRY 
import pickle
import os

def save_model_preproc(model, path):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    
    if not os.path.exists(path) : 
        os.mkdir(path)
    with open(os.path.join(path,"preproc.pkl"),"wb") as f:
        pickle.dump(model, f)

def save_trained_model(model, path):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    
    if not os.path.exists(path) : 
        os.mkdir(path)
    with open(os.path.join(path,"trained_model.pkl"),"wb") as f:
        pickle.dump(model, f)        

def load_model_preproc(path) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(os.path.join(path,"preproc.pkl"),"rb")  as f:
        preproc_model = pickle.load(f)
    return preproc_model

def load_trained_model(path) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    with open(os.path.join(path,"trained_model.pkl"),"rb")  as f:
        trained_model = pickle.load(f)
    return trained_model