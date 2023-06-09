import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from middleware import model_predict

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
    """
    Default route that returns a test message.

    Returns:
        dict: A dictionary containing a test message.
    """
    return {"message": "Test OK"}


@app.post("/predict")
async def run_prediction(payload: dict):
    """
    Endpoint for running a prediction.

    Args:
        payload (dict): A dictionary containing the input data.

    Returns:
        dict: A dictionary containing the prediction result.
    """
    input_string = payload.get("input")
    result = model_predict(input_string)
    print(result)
    return {"result": result}
