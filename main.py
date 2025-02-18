from fastapi import FastAPI
from service import get_predictions_for_week

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predictions/{week}")
async def get_predictions(week: str):
    return get_predictions_for_week(week)
