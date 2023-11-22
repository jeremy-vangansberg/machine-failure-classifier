from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from model_utils import predict_pipeline, load
import numpy as np

app = FastAPI()

class TextIn(BaseModel):
    description: list

class PredictionOut(BaseModel):
    category: list
    probability: list

model, tokenizer = load()

@app.post("/predict", response_model=PredictionOut)
def predict(payload:TextIn):
    print(payload.description)
    category, probabilities = predict_pipeline(payload.description, model, tokenizer)
    probabilities = [probability.item() if isinstance(probability, np.generic) else probability for probability in probabilities]

    return {'category': category,
            'probability' : probabilities}

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=80)
