import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import datetime

def load_dataset (dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dir,
        labels="inferred",       # Automatically label based on folder names
        label_mode="int",        # Labels as integers (can also use 'categorical' or 'binary')
        image_size=(224, 224),   # Resize images to a standard size
        batch_size=16,            # Batch size for training
        shuffle=True,            # Shuffle the dataset
        color_mode="grayscale",        # Color images
    )
    return dataset

# Load the dataset
dataset = load_dataset('dataset')

#  Number of classes
class_names = dataset.class_names
num_classes = len(class_names)

# Convert grayscale images to RGB
def convert_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

dataset = dataset.map(convert_to_rgb)

# Normalize images to [-1, 1] for MobileNetV2
normalize = tf.keras.layers.Rescaling(1./127.5, offset=-1)
dataset = dataset.map(lambda x, y: (normalize(x), y))

# split the dataset into training and validation
train_size = int(0.7 * len(dataset))
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

# Optimize data loading
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Apply data augmentation
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

train_ds = train_ds.map(lambda x, y: (augment(x), y))

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(range(num_classes)),
    y=[label.numpy() for _, label in train_ds.unbatch()]
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

def model_initialize(SHAPE, num_classes, augmentation=None):
    global base_model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=SHAPE)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=SHAPE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

SHAPE = (224, 224, 3)
model = model_initialize(SHAPE, num_classes)

#adding checkpoint
checkpoint = ModelCheckpoint('model/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Learning rate scheduler as a lambda function
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 5e-5 * 0.9 ** (epoch // 10), verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

initial_epochs = 10

# train only the last layer
history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds, callbacks=[checkpoint,early_stopping, tensorboard_callback], class_weight=class_weights) 

# unfreeze layers
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 150

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_ds, epochs=100, initial_epoch=len(history.epoch)
                        , validation_data=val_ds, callbacks=[lr_scheduler, checkpoint, early_stopping, tensorboard_callback], class_weight=class_weights) #class_weight=class_weights


# Evaluate accuracy on validation data
model_loss, model_accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {model_accuracy * 100:.2f}%")


# Save the final model
model.save('model/mobilenet_finetuned_final.keras')


