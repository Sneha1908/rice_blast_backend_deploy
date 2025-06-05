import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

model_path = os.path.join('models', 'rice_leaf_classifier_model_final.keras')
class_names = ['NOT A LEAF', 'NOT A RICE LEAF', 'RICE LEAF']
model = load_model(model_path)

def classify_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[predicted_class]) * 100
    label = class_names[predicted_class]

    return label, confidence
