import numpy as np
import cv2
from tensorflow.keras.models import load_model
# from mtcnn import MTCNN

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
model = load_model('mask_detector.model')

cap = cv2.VideoCapture(0)

def detect_face(img):
    y_pred = np.argmax(model.predict(img.reshape(1,224,224,3)),axis=1)
    return y_pred[0]

while (True):
    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    scaled_img = img/255.0
    predicted_y = detect_face(scaled_img)
    print(predicted_y)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame)
    frame_color = ()
    labels = {1: "Without mask", 0: "With mask"}

    for (x, y, w, h) in faces:
        text_color = (0,0,0)
        text = labels[predicted_y]
        if predicted_y == 0:
            text_color = (0, 255, 0)
        else:
            text_color = (0, 0, 255)
        frame_color = text_color
        cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()