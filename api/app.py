import json
from fastapi import FastAPI, Request
from middleware import model_predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/test")
async def test(payload: dict):
    input_string = payload.get("input")
    return input_string


@app.post("/predict")
async def run_prediction(payload: dict):
    input_string = payload.get("input")
    result = model_predict(input_string)
    print(result)
    return {"result": result}
