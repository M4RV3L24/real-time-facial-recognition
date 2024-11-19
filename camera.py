import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2 as cv
import matplotlib.pyplot as plt
import threading

video = cv.VideoCapture(0)
face_detect = cv.CascadeClassifier('utility/haarcascade_frontalface_default.xml')

model = load_model("model/mobilenet_finetuned_final.keras", compile=False)

# model = load_model("model/mobilenet_finetuned_model_multiclass.keras", compile=False)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# video.set(3, 640)
# video.set(4, 480)

# Load the labels
class_names = open("label.txt", "r").readlines()
print(class_names)


processed_frame = None

def process_frame(frame):
    global best_score, processed_frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]
        crop_image = cv.resize(crop_image, (224, 224), interpolation=cv.INTER_AREA)

        # Make the image a numpy array and reshape it to the models input shape.
        crop_image = np.asarray(crop_image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        crop_image = (crop_image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(crop_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name, end=" - ")
        print("Confidence Score:", str(f"{confidence_score * 100:.2f}"), "%")


        # adding rectangle and text on the frame
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.rectangle(frame, (x, y-40), (x+w, y), (0,255,0), -1)
        cv.putText(frame, class_name+" - "+ str(f"{confidence_score * 100:.2f}") + "%", (x, y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
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
        frame = cv.flip(frame, 1)
        process_frame(frame)
    
    # if processed_frame is not None:
    cv.imshow('frame', processed_frame)
    # else:
    #     cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        pause = not pause

print('Best Score: ', f'{best_score * 100:.2f}', '%')
video.release()
cv.destroyAllWindows()
