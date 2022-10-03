#%%
import numbers
from time import time
import cv2
from cv2 import KeyPoint
import mediapipe as mp
import numpy as np
import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#%%
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def drawing_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

# %%
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("empty camera")
            continue

        image, results = mediapipe_detection(frame, holistic)
        print(results)
        drawing_landmarks(frame, results)
        landmarks = results.pose_landmarks.landmark
        cv2.imshow('frame', frame)

        if cv2.waitKey(5) & 0xff == 27:
            break

cap.release()
cv2.destroyAllWindows()

# %%
pose = []
for res in results.pose_landmarks.landmark:
    test =np.array([res.x, res.y, res.z , res.visibility])
    pose.append(test)

# %%
def landmarksFlatten(results):
    pose = np.array([[res.x, res.y, res.z]for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(21*3)
    
    return pose
    
# %%
data_path = 'test1'
actions = np.array(['1','2'])
for action in actions:
        os.makedirs(os.path.join(data_path, action))

 
# %%
numbers = 40
time_set = [0,1]
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence = 0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("empty camera")
            continue

        image, results = mediapipe_detection(frame, holistic) ##detection to image
        drawing_landmarks(frame, results)                     ##drawing the keypoints        

        if cv2.waitKey(4) & 0xff == 73:
            for number in range(numbers):
                for time in range(time_set[0]):
                    if time == 0:
                        cv2.putText(image, 'start', (3,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                        cv2.imshow('a', image)
                    else:
                        cv2.putText(image, 'saving video number {}'.format(number), (15,12),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('a', image)
            try:
                landmarks = results.pose_landmarks.landmark   
                keypoint = landmarksFlatten(landmarks)
                npy_path = os.path.join(data_path, actions[0])
                np.save(npy_path, keypoint)
            except:
                pass

        cv2.imshow('normal', frame)
        if cv2.waitKey(5) & 0xff == 115:
           break

cap.release()
cv2.destroyAllWindows()
# %%
