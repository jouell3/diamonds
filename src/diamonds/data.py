import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import loguru
import os
from diamonds.params import DATA_PATH

logger = loguru.logger

# Cache dictionary for storing loaded data
cache = {}

# Import other necessary libraries here

def load_data() -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    logger.info("Loading the data ...")
    
   
    if not cache:
        logger.info("Data not found in cache, loading from seaborn ...")
        data = sns.load_dataset('diamonds')
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(DATA_PATH, "raw"), exist_ok=True)
        data.to_csv(os.path.join(DATA_PATH, "raw","diamonds.csv"), index=False)
        cache['diamonds'] = data
    else:
        logger.info("Data found in cache, loading from cache ...")
        data = cache['diamonds']
  
    
    return data

def clean_data(df: pd.DataFrame) -> pd.Series:
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """
    row = len(df)
    def keep_not_null(row) :
        if 0 in row.values : return False
        return True

    df_clean = df[df.apply(keep_not_null, axis=1)]
    logger.info(f"Removed {row - len(df_clean)} rows with null values")
    
    return df_clean

def create_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y

def split_data(df: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    return train_test_split(df, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

