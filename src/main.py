from src.predict import *
from fastapi import FastAPI, UploadFile
from PIL import Image
import redis

app = FastAPI()

r = redis.Redis(host="redis", port=6379)

@app.get("/")
def read_root():
    return {"Hello": "ramy"}


@app.post("/upload")
def upload(image: UploadFile):
    image = Image.open(image.file)
    prediction = predict_(image)
    return {"prediction": prediction}


@app.get("/hits")
def read_root():
    r.incr("hits")
    return {"num of hits", r.get("hits")}