from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from diamonds import model
import uvicorn
    



api = FastAPI()

class InputData(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    price: Optional[float] = None
    x: float
    y: float
    z: float

@api.get("/")
def hello() -> dict[str, str]:
    """A simple endpoint to test the API."""
    return {"message": "Hello, World!"}


@api.post("/predict")
def predict(X: InputData) -> dict[str, float]:
    """Evaluate the model on the given data and return predicted value."""
    diamond_df = pd.DataFrame.from_dict(X.model_dump(), orient="index").T
    print(diamond_df)
    diamond_df.drop(columns=["price"], inplace=True)
    print(diamond_df)
    X_scaled = model.preprocess_data(diamond_df)
    y_pred = model.predict(X_scaled)

    return {"prediction": y_pred[0]}   

@api.post("/predict_multiple")
def predict_multiple(X: list[InputData]) -> list[InputData]:
    """Evaluate the model on the given data and return predicted value."""
    diamond_df = pd.DataFrame([x.model_dump() for x in X])
    print(diamond_df)
    diamond_df.drop(columns=["price"], inplace=True)
    print(diamond_df)
    X_scaled = model.preprocess_data(diamond_df)
    y_pred = model.predict(X_scaled)
    print(list(y_pred))
    diamond_df["price"] = list(y_pred)
    print(diamond_df)
    return diamond_df.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)