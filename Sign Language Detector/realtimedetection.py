import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math

# Load model
model = tf.keras.models.load_model("sign_model.keras")

class_names = ["Hello", "I Love You", "No", "Thanks", "Yes"]  # match your folders

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

imgSize = 224
offset = 20

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:   # process each hand separately
            x, y, w, h = hand['bbox']

            if w == 0 or h == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                aspectratio = h / w

                if aspectratio > 1:
                    k = imgSize / h
                    wCal = int(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = int(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # Prepare input
                imgInput = imgWhite / 255.0
                imgInput = np.expand_dims(imgInput, axis=0)

                prediction = model.predict(imgInput)
                index = np.argmax(prediction)

                label = class_names[index]

                cv2.putText(img, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2)

                cv2.imshow("Processed", imgWhite)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
