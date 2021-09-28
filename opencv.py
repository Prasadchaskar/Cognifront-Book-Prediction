from tensorflow import keras
import numpy as np 
import cv2

# width and height of window
width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

# Read model from book folder
m = keras.models.load_model("book")

while True:
    success,imgOriginal = cap.read()
    img = cv2.resize(imgOriginal,(150,150))
    img = np.array(img)
    img = img/255.0
    cv2.imshow("image",img)
    img = img.reshape((1,150,150,3))
    labels = {  0:'SUCCEEDING IN INTERVIEWS',
            1:'SYMBIOTIC RELATIONSHIP',
            2:'MITIAGATING DEFICIENCIES OF TECHINCAL EDUCATION'}
    cv2.putText(imgOriginal,str(labels[np.argmax(m.predict(img))]),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow("Original",imgOriginal)  
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 # To exit video window press 'q' key