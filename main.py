from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Intel Image Classification classes
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((150, 150))
    img_array = np.array(img) / 255.0

    # Simulate predictions
    scores = np.random.dirichlet(np.ones(len(class_names)))
    predictions = [
        {"class": class_names[i], "score": round(float(scores[i]), 2)}
        for i in range(len(class_names))
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    return {
        "image_id": file.filename,
        "predictions": predictions
    }