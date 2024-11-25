import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, GaussianNoise
from matplotlib import pyplot as plt
import datetime

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

# Normalize images to [-1, 1] for MobileNetV2
normalize = tf.keras.layers.Rescaling(1./255, offset=0)
train_dataset = train_dataset.map(lambda x, y: (normalize(x), y))

# Apply data augmentation
augment = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),
    GaussianNoise(0.05),
])

# Create a summary writer
log_dir = "logs/augmented_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)

# Log augmented images to TensorBoard
def log_augmented_images(dataset, augment, file_writer):
    with file_writer.as_default():
        for images, labels in dataset.take(1):
            augmented_images = augment(images)
            # Rescale images to [0, 255]
            # augmented_images = tf.clip_by_value(augmented_images * 255, 0, 255)
            tf.summary.image("Augmented Images", augmented_images, max_outputs=100, step=0)

log_augmented_images(train_dataset, augment, file_writer)





# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#   for i in range(min(9,len(images))):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()