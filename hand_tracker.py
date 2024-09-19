import cv2 as cv
import time
import mediapipe as mp

cap = cv.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.4,min_tracking_confidence=0.4)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv.flip(img, 1)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    

    results = hands.process(img)

    
    
    if results.multi_hand_landmarks:
        print("landmarks detected")
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,
                                   hand_landmarks,
                                   mphands.HAND_CONNECTIONS,
                                   mp_drawing_styles.get_default_hand_landmarks_style(),
                                   mp_drawing_styles.get_default_hand_connections_style()
                                    )

    cv.imshow('Hand Tracker', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    