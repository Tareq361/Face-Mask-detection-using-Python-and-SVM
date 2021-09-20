import numpy as np
import cv2
import os
import glob
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
path =glob.glob(r"C:\Users\mtare\PycharmProjects\pythonProject\image\not_properly_mask/*.jpg")
faceCascade = cv2.CascadeClassifier(cascPathface)
ImageData = []
count=0
for file in path:
    # Capture frame-by-frame
    image = cv2.imread(file)
    faces = faceCascade.detectMultiScale(image)
    for x,y,w,h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h),(0,255,10), 2)
        face=image[y:y+h,x:x+w, : ]
        face=cv2.resize(face,(50,50))
        ImageData.append(face)
        print(count)

    count=count+1
    cv2.imshow('Face Mask Detector', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
np.save('not_properly_mask',ImageData)