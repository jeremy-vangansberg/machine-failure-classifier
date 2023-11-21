from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from model_utils import predict_pipeline

app = FastAPI()

class TextIn(BaseModel):
    description: list

class PredictionOut(BaseModel):
    category: list

@app.post("/predict", response_model=PredictionOut)
def predict(payload:TextIn):
    print(payload.description)
    preds, probas = predict_pipeline(payload.description)
    return {'category': preds,
            'probability' : probas}

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=80)
