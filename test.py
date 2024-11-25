import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

face_detect = cv.CascadeClassifier('utility/haarcascade_frontalface_default.xml')


def preprocess (image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        crop_image = cv.resize(crop_image, (224, 224), interpolation=cv.INTER_AREA)
    
    return crop_image

image_paths = glob.glob('dataset/nico/*.jpg')
images = [cv.imread(file) for file in image_paths]
process_images = [preprocess(image) for image in images]



# overwrite image in dataset with formatted face 
for i in range(len(images)):
    cv.imwrite(image_paths[i], process_images[i])
