from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Boba Store Predictor",
    description="Classify boba stores as either 1 = open or 0 = not open.",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying boba stores. Choose an index 0 through 9'}

class request_body(BaseModel):
    sample_index: int = None

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    global df_samples
    model_pipeline = joblib.load("boba_model_pipeline.joblib")
    df_samples = pd.read_csv("../../../data/xtrain_sample.csv")



# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    if data.sample_index is not None and (0 <= data.sample_index < len(df_samples)):
        X = df_samples.iloc[[data.sample_index]]
        predictions = model_pipeline.predict_proba(X)
        return {'Predictions': predictions}
    else:
        return {"error": "Enter an index between 0 and 9."}
