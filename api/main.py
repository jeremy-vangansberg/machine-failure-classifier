from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from app.model import predict_pipeline

app = FastAPI()

class TextIn(BaseModel):
    description: str
    url: str

class PredictionOut(BaseModel):
    category: str

@app.post("/predict", response_model=PredictionOut)
def predict(payload:TextIn):
    category_pred = predict_pipeline(payload.description, payload.url)
    return {'category': category_pred}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=80)
