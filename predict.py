import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def predict_image(img_path, model_path='brain_tumor_model.h5'):
    model = load_model(model_path)
    img = Image.open(img_path).resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)
