import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("F:\pycharm-com\haarcascade_frontalface_default.xml")
Eye_cascade = cv2.CascadeClassifier("F:\pycharm-com\haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 3)
        roi_gray = gray[x:y+h, x:x+w]
        roi_color =img[x:y+h, x:x+w]
        eyes = Eye_cascade.detectMultiScale((roi_gray))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (255,0,0), 3)

    cv2.imshow("Video", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyWindow()