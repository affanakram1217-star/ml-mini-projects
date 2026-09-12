# # 
# import mediapipe as mp
# import cv2
# print("Mediapipe OK:", hasattr(mp,"solutions"))
# # print(dir(mp))

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# import time

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)
# offset = 20
# imgSize = 300
# counter = 0

# folder = r"C:\Users\affan\OneDrive\Desktop\Sign Language Detector\Data\Hello"

# while True:
#     success, img = cap.read()
#     if not success:
#         continue

#     hands, img = detector.findHands(img)
#     if hands:
#         for hand in hands:      
#             x, y, w, h = hand['bbox']

#         # ✅ Prevent division error
#         if w == 0 or h == 0:
#             continue

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#         # ✅ FIXED CROP (no out-of-bounds)
#         y1 = max(0, y - offset)
#         y2 = min(img.shape[0], y + h + offset)
#         x1 = max(0, x - offset)
#         x2 = min(img.shape[1], x + w + offset)

#         imgCrop = img[y1:y2, x1:x2]

#         # ✅ SAFETY CHECK (prevents crash)
#         if imgCrop is None or imgCrop.size == 0:
#             print("Empty crop, skipping frame")
#             continue

#         aspectratio = h / w

#         if aspectratio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize

#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize

#         cv2.imshow('ImageCrop', imgCrop)
#         cv2.imshow('ImageWhite', imgWhite)

#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)

#     if key == ord('s'):
#         counter += 1
#         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
#         print(counter)

#     if key==ord('q'):
#         break;