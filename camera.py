import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2 as cv
import matplotlib.pyplot as plt
# import threading
from facenet_pytorch import MTCNN 

video = cv.VideoCapture(0)
# face_detect = cv.CascadeClassifier('utility/haarcascade_frontalface_default.xml')
detector = MTCNN('cuda:0')
model = load_model("model/mobilenet_finetuned_final.keras", compile=False)

# model = load_model("model/mobilenet_finetuned_model_multiclass.keras", compile=False)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Load the labels
class_names = [line.strip() for line in open("label.txt", "r").readlines()]

print(class_names)


processed_frame = None
smoothed_confidence_score = 0
alpha = 0.1  # Smoothing factor for EMA

# Create a named window
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 640, 480)


def process_frame(frame):
    global best_score, processed_frame, smoothed_confidence_score
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # faces = face_detect.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

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
            crop_image = cv.resize(crop_image, (224, 224), interpolation=cv.INTER_AREA)
            crop_image = cv.cvtColor(crop_image, cv.COLOR_BGR2GRAY)
            crop_image = cv.cvtColor(crop_image, cv.COLOR_GRAY2RGB)


            # Make the image a numpy array and reshape it to the models input shape.
            crop_image = np.asarray(crop_image, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            crop_image = (crop_image / 127.5) - 1

            # Predicts the model
            prediction = model.predict(crop_image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Smooth the confidence score using EMA
            smoothed_confidence_score = alpha * confidence_score + (1 - alpha) * smoothed_confidence_score
            

            # Print prediction and confidence score
            print("Class:", class_name, end=" - ")
            print("Confidence Score:", str(f"{smoothed_confidence_score * 100:.2f}"), "%")


            # adding rectangle and text on the frame
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.rectangle(frame, (x, y-40), (x+w, y), (0,255,0), -1)
            cv.putText(frame, class_name+" - "+ str(f"{smoothed_confidence_score * 100:.2f}") + "%", (x, y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            best_score = max(best_score, confidence_score)
        
    processed_frame = frame


# Initialize variables
best_score = 0
frame_skip = 0  # Process every 2nd frame
frame_count = 0
pause = False

while True:

    if not pause:
        ret, frame = video.read()
        frame = cv.flip(frame, 1)
        process_frame(frame)
        # threading.Thread(target=process_frame, args=(frame.copy(),)).start()
    
    
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

    else:
        cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        pause = not pause

print('Best Score: ', f'{best_score * 100:.2f}', '%')
video.release()
cv.destroyAllWindows()
