import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(_ROOT, "data")
MODEL_PATH = os.path.join(_ROOT, "models")

# MLFLOW CONFIG
MODEL_REGISTRY = "model"
MODEL_NAME = "trained_model"
PREPROD_MODEL_NAME = "preproc"
ALIAS = "prod"

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")
MODEL_MLFLOW_URI = os.environ.get("MODEL_MLFLOW_URI", "http://localhost:5000")




