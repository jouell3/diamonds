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


def create_model(model_name="KNN") -> BaseEstimator:
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
        return LinearRegression()
    elif model_name == "random_forest":
        return RandomForestRegressor(n_estimators=500)
    elif model_name == "KNN":
        return KNeighborsRegressor()    
    elif model_name == "SVM":
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
    
    return preprocessor

def preprocess_data(model: BaseEstimator, df: pd.DataFrame)  -> pd.DataFrame:
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
    return model.transform(df)

def train_model(model: BaseEstimator, X_train, y_train):
    model_trained = model.fit(X_train, y_train)
    return model_trained
    

def evaluate_model(model, X_test, y_true) -> dict[str, float]: 
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    r2  = r2_score(y_true,y_pred)
    mape = mean_absolute_percentage_error(y_true,y_pred) 
    scores = {"mape": mape, "mae":mae,"mse":mse,"r2":r2}
    return scores

def predict(model, X):
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
    return model.predict(X)
