from sklearn.metrics import confusion_matrix
import pickle as pkl
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW  # Import the custom optimizer

# Load the model with the custom optimizer specified
model = tf.keras.models.load_model('/content/model.h5', custom_objects={'AdamW': AdamW})

# Load and preprocess a single image
img_path = '/content/drive/MyDrive/plant/val/Brown_rust/Brown_rust036.jpg'
img = image.load_img(img_path, target_size=(100, 100))  # Resize as per model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize if necessary

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)

