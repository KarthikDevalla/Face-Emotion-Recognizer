import cv2
from keras.models import load_model
import numpy as np

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)
emotions=['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sad','Surprise']
model=load_model('face_resnet.h5')

while True:
    frame_read,frame = webcam.read()
    greyscale_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_cordinates = trained_data.detectMultiScale(greyscale_frame)
    for (x,y,w,h) in face_cordinates:
        resize=cv2.resize(frame,(224,224))
        normalize=resize/255.0
        normalize=np.reshape(normalize,(1,224,224,3))
        pred=model.predict(normalize)
        # print(pred)
        label=np.argmax(pred,axis=1)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,emotions[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Emotion Recognizer",frame)
    cv2.waitKey(1)