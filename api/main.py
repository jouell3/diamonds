from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from diamonds import model
import uvicorn

from variable_model import InputData, OutputData
    
api = FastAPI()
templates = Jinja2Templates(directory="api/templates")
pipeline = model.load_model("full_pipeline")

@api.get("/")
def hello(request: Request):
    """Render the landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@api.post("/predict")
def predict(X: InputData) -> OutputData:
    """Evaluate the model on the given data and return predicted value."""
    diamond_df = pd.DataFrame.from_dict(X.model_dump(), orient="index").T
    #diamond_df.drop(columns=["price"], inplace=True)
    #X_scaled = model.preprocess_data(diamond_df)
    y_pred = model.predict(pipeline, diamond_df)
    diamond_df["price"] = list(y_pred)
    return OutputData(**diamond_df.to_dict(orient="records")[0])

@api.post("/predict_multiple")
def predict_multiple(X: list[InputData]) -> list[OutputData]:
    """Evaluate the model on the given data and return predicted value."""
    diamond_df = pd.DataFrame([x.model_dump() for x in X])
    #diamond_df.drop(columns=["price"], inplace=True)
    #X_scaled = model.preprocess_data(diamond_df)
    y_pred = model.predict(pipeline, diamond_df)
    diamond_df["price"] = list(y_pred)
    print(diamond_df)
    return [OutputData(**row) for row in diamond_df.to_dict(orient="records")]

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)