import os
from pathlib import Path


def _find_project_root() -> Path:
	"""Return a project root that contains the local data/models folders.

	When this package is installed in site-packages (e.g. inside Docker), the
	source file location is no longer the project root. In that case we fall
	back to the current working directory used to run the API.
	"""
	package_root = Path(__file__).resolve().parents[2]
	if (package_root / "models").exists() and (package_root / "data").exists():
		return package_root
	return Path.cwd()


_ROOT = _find_project_root()

MODEL_PATH = os.environ.get("MODEL_PATH", str(_ROOT / "models"))
DATA_PATH = os.environ.get("DATA_PATH", str(_ROOT / "data"))

# MLFLOW CONFIG
MODEL_REGISTRY = "local"
PIPELINE_NAME = "full_pipeline"
MODEL_NAME = "trained_model"
PREPROD_MODEL_NAME = "preproc"
ALIAS = "prod"

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")
MODEL_MLFLOW_URI = os.environ.get("MODEL_MLFLOW_URI", "http://localhost:5000")




