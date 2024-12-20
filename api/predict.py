from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()

# Load the model
model = load_model("./Electronic-Component-Detector-mod-best.keras")

# Define class names
class_names = {
    0: 'arduino',
    1: 'battery',
    2: 'DCmotor',
    3: 'DHT-11',
    4: 'ESP8266',
    5: 'LCD',
    6: 'Loadcell',
    7: 'RFID',
    8: 'Tiva',
    9: 'Ultrasonic',
}

# Prediction function
def predict_image(model, img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    data = np.asarray(img) / 255.0
    probs = model.predict(np.expand_dims(data, axis=0))
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    return top_prob, top_pred

# API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    img = Image.open(file.file)
    top_prob, top_pred = predict_image(model, img)
    return {"confidence": round(top_prob * 100, 2), "class": top_pred}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
