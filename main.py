import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("mobmodel.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_img_path = input("Enter the path to the input image: ")

input_img = cv2.imread(input_img_path)
if input_img is None:
    print("Invalid image path. Please provide a valid path to the image.")
    exit(0)
    
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

for (x, y, w, h) in faces_detected:
    cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=7)
    roi_gray = gray_img[y:y + w, x:x + h]
    roi_gray = cv2.resize(roi_gray, (224, 224))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]
    cv2.putText(input_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

resized_img = cv2.resize(input_img, (1000, 700))
cv2.imshow('Facial emotion analysis', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
