from fastapi import FastAPI
import uvicorn
import sys
import os
import subprocess
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from TextSummerizer.pipeline.prediction import PredictionPipeline


text:str = "What is Text Summerization?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        return Response("Training successful !!")
    except subprocess.CalledProcessError as e:
        # Catches specifically if main.py crashes or fails
        return Response(f"Training failed with error code: {e.returncode}")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        return Response(f"Error Occurred! {e}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
