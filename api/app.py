from fastapi import FastAPI, Request
from middleware import model_predict

app = FastAPI()


@app.post("/predict")
async def run_prediction(request: Request):
    input_string = await request.body()    
    result = model_predict(input_string)
    return {"result": result}