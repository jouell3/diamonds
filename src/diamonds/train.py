import pandas as pd





from diamonds.data import load_data, clean_data, create_X_y, split_data
from diamonds.model import create_model, create_preproc, train_model, evaluate_model, predict, preprocess_data

def train(
    model_name: str = "KNN",
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
    # 1) Data
  
    # 2) Model + preprocessing
 
    # 3) Evaluation
  
    # 4) Persistence
    
    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    print("="*60)
    print("Shape of teh training and testing datasets")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("="*60)
    preproc = create_preproc(X_train)
    X_train_scaled = preprocess_data(preproc, X_train)
    X_test_scaled = preprocess_data(preproc, X_test)
    model_ = create_model(model_name)
    model_ = train_model(model_, X_train_scaled, y_train)
    print("="*60)
    print("Evaluattion of the model")
    print(evaluate_model(model_, X_test_scaled, y_test))
    print("="*60)

if __name__ == "__main__":
    train()

