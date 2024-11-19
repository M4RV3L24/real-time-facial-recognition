import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import json
from matplotlib import pyplot as plt

# Load the dataset
dataset_dir = "dataset"
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels="inferred",       # Automatically label based on folder names
    label_mode="int",        # Labels as integers (can also use 'categorical' or 'binary')
    image_size=(224, 224),   # Resize images to a standard size
    batch_size=16,            # Batch size for training
    shuffle=True,            # Shuffle the dataset
    color_mode="grayscale",        # Color images
)
#  Number of classes
class_names = train_dataset.class_names
num_classes = len(class_names)

# Convert grayscale images to RGB
def convert_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

train_dataset = train_dataset.map(convert_to_rgb)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(min(9,len(images))):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()