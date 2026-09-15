from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from kitty3000_ml.predict import predict, available_models, default_model
from kitty3000_ml.labels import LABELS

app = FastAPI()

# MK: let the web front end (a browser page on another domain) call this API.
# MK: public, read-only inference API, so any origin is fine for the demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {"message": "Kitty3000 API is running", "docs": "/docs", "health": "/health", "models": "/models", "predict": "POST /predict?model=panns"}


@app.get("/health")
def health():
    return {"status": "API healthy", "models": available_models(), "default_model": default_model(), "labels": LABELS}


@app.get("/models")
def models():
    return {"models": available_models(), "default": default_model()}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), model: str = None):
    if model is None:
        model = default_model()
    if model not in available_models():
        raise HTTPException(status_code=400, detail="model '" + model + "' is not available, we have: " + str(available_models()))
    audio_bytes = await file.read()
    result = predict(audio_bytes, model)
    result["filename"] = file.filename
    return result


@app.post("/predict/{model}")
async def predict_by_path(model: str, file: UploadFile = File(...)):
    return await predict_endpoint(file, model)
