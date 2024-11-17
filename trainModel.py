import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import json

# Set dataset directory
dataset_dir = "dataset"

# Load the dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels="inferred",       # Automatically label based on folder names
    label_mode="int",        # Labels as integers (can also use 'categorical' or 'binary')
    image_size=(224, 224),   # Resize images to a standard size
    batch_size=16,            # Batch size for training
    shuffle=True,            # Shuffle the dataset
)
#  Number of classes
class_names = dataset.class_names
num_classes = len(class_names)

# Normalize images to [-1, 1] for MobileNetV2
normalize = tf.keras.layers.Rescaling(1./127.5, offset=-1)
dataset = dataset.map(lambda x, y: (normalize(x), y))

# split the dataset into training and validation
train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

# Optimize data loading
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Apply data augmentation
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)
])
train_ds = train_ds.map(lambda x, y: (augment(x), y))

# Optimize data loading
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(range(num_classes)),
    y=[label.numpy() for _, label in dataset.unbatch()]
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")


# Load the MobileNet model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

#Add new layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#adding checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('model/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

# train only the last layer
model.fit(train_ds, epochs=10, batch_size=16, validation_data=val_ds, callbacks=[checkpoint], class_weight=class_weights)

# unfreeze layers
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=50, batch_size=16, validation_data=val_ds, callbacks=[checkpoint, early_stopping], class_weight=class_weights)

# Save the training history
with open('model/training_history.json', 'w') as f:
    json.dump(history.history, f)


# Evaluate accuracy on validation data
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the final model
model.save('model/mobilenet_finetuned_final.keras')
