import numpy as np
import cv2
from tensorflow.keras.models import load_model
# from mtcnn import MTCNN

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
model_vgg = load_model('vgg_mask_detector.model')
# detector = MTCNN()

cap = cv2.VideoCapture(0)

def detect_face(img):
    y_pred_vgg = np.argmax(model_vgg.predict(img.reshape(1,224,224,3)),axis=1)
    return y_pred_vgg[0]

while (True):
    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    scaled_img = img/255.0
    predicted_y_vgg = detect_face(scaled_img)
    print(predicted_y_vgg)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray)

    faces = detector.detect_faces(img)

    frame_color = ()
    labels = {1: "Without mask", 0: "With mask"}

    for single_face in faces:
        x,y,w,h = single_face['box']
        text_color = (0,0,0)
        text = labels[predicted_y_vgg]
        if predicted_y_vgg == 0:
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