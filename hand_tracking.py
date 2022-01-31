import mediapipe
import cv2

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
cap = cv2.VideoCapture('/content/drive/MyDrive/Colab Notebooks/VID_20220128_195007.mp4')

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
     while True:
           ret, frame = cap.read()
           results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
           
           if results.multi_hand_landmarks != None:
              for handLandmarks in results.multi_hand_landmarks:
                  drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
           
           cv2.imshow('Test hand', frame)
           key = cv2.waitKey(1) & 0xFF
           
           if key == ord("q"):
              break

cv2.destroyAllWindows()
capture.release()
