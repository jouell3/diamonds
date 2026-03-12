
from pydantic import BaseModel




class InputData(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

class OutputData(InputData):
    price: float
