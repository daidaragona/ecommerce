import json
from fastapi import FastAPI, Request
from middleware import model_predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def run_prediction(request: Request):
    input_string = await request.body()
    print(input_string)
    result = model_predict(input_string)
    return {"result": result}
