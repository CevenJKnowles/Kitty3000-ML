from fastapi import FastAPI, File, UploadFile

from kitty3000_ml.predict import predict # MK: predict.py contains the main function that will be called by the API, it will call the model and return the label with the highest score and the scores for each label
from kitty3000_ml.labels import LABELS

app = FastAPI()

@app.get("/") # MK: root endpoint, so the base URL shows something instead of 404
def index():
    return {"message": "Kitty3000 API is running", "docs": "/docs", "health": "/health", "predict": "POST /predict"}

@app.get("/health")
def health():
    return {"status": "API healthy", "model": "dummy-test-model", "labels": LABELS} # MK: return the health status of the API, the model name and the labels

@app.post("/predict") # MK: This is the main endpoint that will be called by the client, it will call the predict function and return the label with the highest score and the scores for each label
async def predict_endpoint(file: UploadFile = File(...)): # MK: use async because file might be large and we don't want to block the server
    audio_bytes = await file.read()                       # MK: read the file as bytes, this will be passed to the predict function
    result = predict(audio_bytes)                         # MK: call the predict function and get the result, which is a dictionary with the label with the highest score and the scores for each label
    result["filename"] = file.filename                    # MK: add the filename to the result, so the client knows which file was processed
    return result
