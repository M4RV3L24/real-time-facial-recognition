import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Initialize MTCNN face detector and InceptionResnetV1 model
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)  
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def save_embeddings(embeddings, labels, filepath="embeddings.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump((embeddings, labels), f)

def load_embeddings(filepath="embeddings.pkl"):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# Function to extract embeddings from an image
def get_embedding(image_path):
    # Read the image
    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Detect faces in the image
    faces, _ = mtcnn.detect(img_rgb)

    if faces is not None:
        x, y, x2, y2 = [int(coord) for coord in faces[0]]
        w, h = x2 - x, y2 - y
        face_img = img_rgb[y:y+h, x:x+w]


        # Resize the face image to the expected input size (160x160)
        face_img = cv.resize(face_img, (160, 160))

        # Convert the face image to a PyTorch tensor
        face_img = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Normalize the face image
        face_img = (face_img - 127.5) / 128.0

        # Get embeddings for the face image
        embeddings = model(face_img)
        return embeddings.detach().cpu().numpy()
    else:
        return None

        
# Function to compare the input image with dataset
def compare_face(input_image, dataset_embeddings, dataset_labels, threshold=0.5):
    # Get embedding for input image
    input_embedding = get_embedding(input_image)

    if input_embedding is not None:
        input_embedding = input_embedding.flatten()

        # Compare with each image in the dataset using cosine similarity
        similarities = []
        for embedding, label in zip(dataset_embeddings, dataset_labels):
            similarity = cosine_similarity([input_embedding], [embedding.flatten()])[0][0]
            euclidean_dist = euclidean_distances([input_embedding], [embedding.flatten()])[0][0]
            weighted_score = 0.7 * similarity + 0.3 * (1 - euclidean_dist / np.sqrt(2))  # Weighted average
            if weighted_score > threshold:
                similarities.append((label, weighted_score)) 

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the most similar class
        return similarities[0]

    else:
        return None

# Function to load dataset embeddings
def load_dataset_embeddings(dataset_path):
    if (os.path.exists("utility/embeddings.pkl")):
        return load_embeddings("utility/embeddings.pkl")
    else:
        embeddings = []
        labels = []

        for label in os.listdir(dataset_path):
            img_path = os.path.join(dataset_path, label)
            embedding = get_embedding(img_path)

            if embedding is not None:
                embeddings.append(embedding)
                label = os.path.splitext(label)[0].replace("_", " ")
                labels.append(label)

        return embeddings, labels


# # Load dataset embeddings
dataset_path = 'dataset2/'  # Path to the folder containing your images

dataset_embeddings, dataset_labels = load_dataset_embeddings(dataset_path)
save_embeddings(dataset_embeddings, dataset_labels, "utility/embeddings.pkl")
 

