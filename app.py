from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from joblib import load
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import List
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

blur_model = tf.keras.models.load_model("ML.h5")
image_classification_model = tf.keras.models.load_model("my_classification_model.h5")

def preprocess_image(file, target_size=(224, 224)):
    try:
        img = Image.open(file.file)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def preprocess_image_for_classification(img, target_size=(224, 224)):
    img = Image.open(file.file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_and_check(image_data, threshold=0.5):
    print("predicting")
    prediction = blur_model.predict(image_data)
    class_idx = np.argmax(prediction)
    classes = ['Defocused Blur','Motion Blur','Sharp']
    predicted_class = classes[class_idx]
    confidence = prediction[0][class_idx]

    print(predicted_class)
    print(confidence)

    if predicted_class not in ['Defocused Blur', 'Motion Blur'] and confidence > threshold:
        return "Accepted", predicted_class, confidence
    else:
        return "Rejected", predicted_class, confidence


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")


        img = preprocess_image(file)
        if img is None:
            raise HTTPException(status_code=500, detail="Error preprocessing image.")
        

        # prediction from the first model 
        blur_prediction = predict_and_check(img)

        if blur_prediction[0] == 'Rejected':
            return JSONResponse(content={"error": "Image is blurred"}, status_code=400)
        

        # prediction from the second model 
        prediction = image_classification_model.predict(img)
        predicted_class_index = np.argmax(prediction)

        # Map predicted class index to class label
        # potholes - RDA
        # broken_pipeline - Water Board
        # drainage - Council
        class_labels = ['potholes', 'broken_pipeline', 'drainage']
        predicted_class_label = class_labels[predicted_class_index]

        authority_mapping = {
            'potholes': 'ROAD',
            'broken_pipeline': 'WATER',
            'drainage': 'URBAN'
        }

        authority = authority_mapping.get(predicted_class_label, 'Unknown')

        print("Predicted class:", predicted_class_label)
        print("Authority:", authority)

        return JSONResponse(content={"authority": authority}, status_code=200)

    except HTTPException as http_exception:
        print(http_exception)
        return JSONResponse(content={"error": http_exception.detail}, status_code=http_exception.status_code)

    except Exception as e:
        print(e)
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

