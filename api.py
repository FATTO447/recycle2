### FastAPI (api.py)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Circular Resource AI API")

# Load your model once
model = load_model("my_model.keras")

class_names = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

recommendations_dict = {
    "battery": [
        "Keep used batteries in a sealed container.",
        "Do not throw them in household trash.",
        "Take them to an electronic or battery recycling point."
    ],
    "biological": [
        "Compost organic waste like food scraps and leaves.",
        "Avoid mixing biological waste with plastics.",
        "Use closed compost bins to prevent odor."
    ],
    "brown-glass": [
        "Rinse and separate from other glass colors.",
        "Do not mix brown with clear or green glass.",
        "Deliver to brown-glass recycling bins."
    ],
    "cardboard": [
        "Flatten boxes before recycling to save space.",
        "Keep them dry and free of grease or food stains.",
        "Reuse for storage or crafts if clean."
    ],
    "clothes": [
        "Donate wearable clothes to charity.",
        "Repurpose old fabrics into cleaning rags.",
        "Avoid burning textile waste."
    ],
    "green-glass": [
        "Rinse and separate from clear or brown glass.",
        "Reuse as vases or jars if safe.",
        "Deliver to glass recycling centers."
    ],
    "metal": [
        "Rinse cans and avoid mixing metals.",
        "Flatten or crush large cans.",
        "Donate scrap metal to recycling facilities."
    ],
    "paper": [
        "Separate dry and wet paper.",
        "Reuse for crafts, packaging, or notes.",
        "Recycle if too torn or dirty."
    ],
    "plastic": [
        "Clean and remove labels or caps.",
        "Use for DIY crafts like planters.",
        "Make eco-bricks with clean soft plastics."
    ],
    "shoes": [
        "Donate wearable shoes to NGOs.",
        "Repurpose old ones for planters or décor.",
        "Recycle materials at textile collection points."
    ],
    "trash": [
        "Avoid burning mixed trash.",
        "Try to separate recyclable materials first.",
        "Dispose of non-recyclables responsibly."
    ],
    "white-glass": [
        "Rinse and separate from colored glass.",
        "Handle with care to avoid breakage.",
        "Recycle at glass drop-off stations."
    ]
}


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)
    preds = model.predict(img_array)
    top_indices = preds[0].argsort()[-3:][::-1]

    top_predictions = [
        {
            "label": class_names[i],
            "confidence": float(preds[0][i]),
            "recommendations": recommendations_dict.get(class_names[i], [])
        }
        for i in top_indices
    ]
    return JSONResponse({"top_predictions": top_predictions})
