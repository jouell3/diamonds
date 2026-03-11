from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import time
from diamonds.registry import load_model
import loguru


logger = loguru.logger

#Pour retirer les messages d'erreur Pylance quand pas pertinent
from typing import cast

def create_model(model_name="random_forest") -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "KNN", "random_forest", "linear_regressor", "SVM)

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """
    if model_name == "linear_regressor":
        logger.info("Creating a linear regression model")
        return LinearRegression()
    elif model_name == "random_forest":
        logger.info("Creating a random forest regression model")
        return RandomForestRegressor(max_depth= 20, min_samples_leaf= 2, min_samples_split=2,n_estimators=100)
    elif model_name == "KNN":
        logger.info("Creating a KNN regression model")
        return KNeighborsRegressor()    
    elif model_name == "SVM":
        logger.info("Creating a SVM regression model")
        return SVR()
    

def create_preproc(df: pd.DataFrame) -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    cat_pipe = Pipeline(
    [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
      ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])
    num_pipe = Pipeline(
    [("knn_imp", KNNImputer(n_neighbors=5))
     ,("scaler", StandardScaler())
      ])
    preprocessor = ColumnTransformer(
    [("numeric",num_pipe, make_column_selector(dtype_include="number"))
    ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
      ]).set_output(transform="pandas")
    
    preprocessor.fit(df)
    logger.info("Preprocessing pipeline created and fitted on the training data")
    
    return preprocessor

def preprocess_data(df: pd.DataFrame)  -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """
    model = load_model("preproc")
    cleaned_df = model.transform(df)
    logger.info("Data preprocessed using the preprocessing pipeline")
    logger.info(f"Shape of the preprocessed data: {cleaned_df.shape} compared to the original data: {df.shape}")
    return cleaned_df

def train_model(model: BaseEstimator, X_train, y_train):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    total_time = time.perf_counter() - start
    logger.info(f"Time to train the model: {total_time:.3f} sec")
    return model
    

def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]: 
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) 
    scores = {"mape": mape, "mae":mae,"mse":mse,"r2":r2}
    logger.info("Model evaluation completed")
    logger.info(f"Model scores: {scores}")
    return scores

def predict(X):
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    model = load_model("trained_model")
    logger.info("Model properly loaded from local model registry")
    y_pred = model.predict(X)
    logger.info("Predictions made using the trained model")
    return y_pred

def run_model(X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Run the model using mkflow production trained model"""
    
    model = load_model("trained_model")
    logger.info("Model properly loaded from local model registry")
    y_pred = predict(model, X_test)
    scores = evaluate_model(y_test, y_pred)
    
    logger.info("Model evaluation scores logged to mlflow")
    return y_pred, scores

def run_model_mkflow(X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Run the model using mkflow production trained model"""
    
    model = load_model()
    logger.info("Model properly loaded from mlflow model registry")
    y_pred = predict(model, X_test)
    scores = evaluate_model(y_test, y_pred)
    

    logger.info("Model evaluation scores logged to mlflow")
    return y_pred, scores