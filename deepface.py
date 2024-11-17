import cv2 as cv

import face_recognition as face_rec
import numpy as np

video = cv.VideoCapture(0)

ref_img = cv.imread('marvel.jpeg')
rgb_img = cv.cvtColor(ref_img, cv.COLOR_BGR2RGB)
ref_img_encoding = face_rec.face_encodings(ref_img)[0]

known_encoding = [
    ref_img_encoding
]

known_label = [
    'Marvel'
]


while True:
    ret, frame = video.read()
    img_encoding = face_rec.face_encodings(rgb_frame)[0]

    resized_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
     # Find all the faces and face encodings in the current frame of video
    face_locations = face_rec.face_locations(rgb_frame)
    face_encodings = face_rec.face_encodings(rgb_frame, face_locations)

    face_name = []
    for face_encoding in face_encodings:
        matches = face_rec.compare_faces(known_encoding, face_encoding)
        name = 'Unknown'

        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_label[first_match_index]

        distances = face_rec.face_distance(known_encoding, face_encoding)
        best_match_index = np.argmin(distances)
        if matches[best_match_index]:
            name = known_label[best_match_index]

        face_name.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_name):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

         # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv.destroyAllWindows()



