# predict_webcam.py
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("object_detector.h5")
class_names = ["pen", "glasses", "eraser", "notebook"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    label = f"{class_names[class_index]} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Object Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
