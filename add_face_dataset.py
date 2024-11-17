import numpy as np
import cv2 as cv

video = cv.VideoCapture(0)
face_detect = cv.CascadeClassifier('utility/haarcascade_frontalface_default.xml')

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

video.set(3, 640)
video.set(4, 480)

i = 0
best_score = 0
while True:
    _, frame = video.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]
        crop_image = cv.resize(crop_image, (224, 224), interpolation=cv.INTER_AREA)
        gray_image = cv.cvtColor(crop_image, cv.COLOR_BGR2GRAY)
        cv.imshow("Face", gray_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.imwrite(f"dataset/marvel/face_{i}.jpg", gray_image)
            i += 1

    if cv.waitKey(1) & 0xff == 27:
        break


video.release()
cv.destroyAllWindows()
