import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2 as cv
import matplotlib.pyplot as plt
import json

# show model graph
# model = load_model("model/mobilenet_finetuned_final.keras", compile=False)
# model.summary()


# Load the training history
with open('model/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([history['initial_epochs']-1,history['initial_epochs']-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([history['initial_epochs']-1,history['initial_epochs']-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()