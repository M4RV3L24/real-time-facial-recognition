import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

video = cv.VideoCapture(0)
face_detect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

video.set(3, 640)
video.set(4, 480)

faces_data = [] 
faces_labels = []


i = 0
while True:
    ret, frame = video.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]
        crop_image = cv.resize(crop_image, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(crop_image)
            faces_labels.append(1)
        i+=1
        cv.putText(frame, str(len(faces_data)), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == ord('q') or len(faces_data) >= 100:
        break

video.release()
cv.destroyAllWindows()
