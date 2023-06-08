import json
from fastapi import FastAPI, Request
from middleware import model_predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Test OK"}


@app.post("/predict")
async def run_prediction(payload: dict):
    input_string = payload.get("input")
    result = model_predict(input_string)
    print(result)
    return {"result": result}
