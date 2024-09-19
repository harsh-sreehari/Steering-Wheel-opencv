import cv2 as cv
import time
import mediapipe as mp

cap = cv.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_draw = mp.solutions.drawing_utils



while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    

    results = hands.process(imgRGB)
    cv.imshow('Hand Tracker', img)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
    