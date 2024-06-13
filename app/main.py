from fastapi import FastAPI
from model import predict

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict_injury(input):
    result = predict(input)
    return {"answer":result}