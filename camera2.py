import numpy as np  # TensorFlow is required for Keras to work
import cv2 as cv
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1 
import pickle
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import threading
import queue


# Set the device to GPU if availqable and load the MTCNN face detector
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
video = cv.VideoCapture(0)
detector = MTCNN(device, keep_all=True, post_process=True)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def save_embeddings(embeddings, labels, filepath="embeddings.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump((embeddings, labels), f)

def load_embeddings(filepath="embeddings.pkl"):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def compare_embeddings(input_embedding, dataset_embeddings, dataset_labels):
    # Calculate the cosine similarity between the input embedding and all dataset embeddings

    similarities = []
    for embedding, label in zip(dataset_embeddings, dataset_labels):
        similarity = cosine_similarity([input_embedding], [embedding.flatten()])[0][0]
        similarities.append((label, similarity))


    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[0]

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

video.set(cv.CAP_PROP_FPS, 15)
video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if os.path.exists("utility/embeddings-vgg.pkl"):
    dataset_embeddings, dataset_labels = load_embeddings("utility/embeddings-vgg.pkl")


processed_frame = None
smoothed_confidence_score = 0
alpha = 0.1  # Smoothing factor for EMA

# Create a named window
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 640, 480)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_frame(frame):
    global best_score, processed_frame, smoothed_confidence_score


    faces, probs = detector.detect(frame)
    if faces is not None:
        for i, face in enumerate(faces):
            # Extract the coordinates of the face
            x, y, x2, y2 = [int(coord) for coord in face]
            w, h = x2 - x, y2 - y  # Calculate width and height
            crop_image = frame[y:y+h, x:x+w]

            if crop_image.size == 0:
                print(f"Warning: Empty face crop at coordinates {x}, {y}, {x2}, {y2}")
                continue  # Skip this face
            crop_image = cv.resize(crop_image, (160, 160), interpolation=cv.INTER_AREA)
            # Resize the face image to the expected input size (160x160)
            
            # Convert the face image to a PyTorch tensor
            face_img = torch.tensor(crop_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Normalize the face image
            face_img = (face_img - 127.5) / 128.0

            # Get embeddings for the face image
            input_embeddings = model(face_img).detach().cpu().numpy().flatten()
            result = compare_embeddings(input_embeddings, dataset_embeddings, dataset_labels)
            if result[1] < 0.6:
                class_name = "Unknown"
            else :
                class_name = result[0]
            confidence_score = ((result[1]+1)/2)*100
            # class_name = result[0]

            # Smooth the confidence score using EMA
            smoothed_confidence_score = alpha * confidence_score + (1 - alpha) * smoothed_confidence_score
            # smoothed_confidence_score = confidence_score
            

            # Print prediction and confidence score
            print("Class:", class_name, end=" - ")
            print("Confidence Score:", str(f"{smoothed_confidence_score:.2f}"))


            # adding rectangle and text on the frame
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.rectangle(frame, (x, y-40), (x+w, y), (0,255,0), -1)
            cv.putText(frame, class_name+" - "+ str(f"{smoothed_confidence_score:.2f}"), (x, y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            best_score = max(best_score, confidence_score)
        
    processed_frame = frame


# Initialize variables
best_score = 0
frame_skip = 2  # Process every 2nd frame
frame_count = 0
pause = False


while True:

    if not pause:
        ret, frame = video.read()
        
        frame_count += 1
        if frame_count % frame_skip == 0:
            frame = cv.flip(frame, 1)
            process_frame(frame)

    if processed_frame is not None:
    
        # Get the current size of the window
        if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) >= 1:
            window_rect = cv.getWindowImageRect('frame')
            window_width = window_rect[2]
            window_height = window_rect[3]

            # Calculate the aspect ratio of the original frame
            aspect_ratio = processed_frame.shape[1] / processed_frame.shape[0]

            # Calculate the new dimensions while maintaining the aspect ratio
            if window_width / window_height > aspect_ratio:
                new_height = window_height
                new_width = int(window_height * aspect_ratio)
            else:
                new_width = window_width
                new_height = int(window_width / aspect_ratio)
            
            # Resize the frame to fit the window size while maintaining the aspect ratio
            if new_width > 0 and new_height > 0:
                resized_frame = cv.resize(processed_frame, (new_width, new_height))

                # Create a black canvas with the size of the window
                canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

                # Calculate the position to center the resized frame on the canvas
                x_offset = (window_width - new_width) // 2
                y_offset = (window_height - new_height) // 2

                # Place the resized frame on the canvas
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

                cv.imshow('frame', canvas)

    # else:
    #     cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        pause = not pause

print('Best Score: ', f'{best_score:.2f}', '%')

video.release()
cv.destroyAllWindows()

