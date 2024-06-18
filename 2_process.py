import cv2
import os
import mediapipe as mp
import numpy as np
import pickle
import csv

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

data_path = os.path.join('.', 'data')

data = []
labels = []

for subfolder in os.listdir(data_path):
    subfolder_path = os.path.join(data_path, subfolder)
    
    if os.path.isdir(subfolder_path):
        for class_dir in os.listdir(subfolder_path):
            class_dir_path = os.path.join(subfolder_path, class_dir)
            
            if os.path.isdir(class_dir_path):
                print(class_dir)
                for img in os.listdir(class_dir_path):
                    if img.endswith('.JPEG'):
                        data_aux = []
                        print(class_dir, img)
                        img_path = os.path.join(class_dir_path, img)
                        img_array = cv2.imread(img_path)
                        
                        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        
                        with mp_holistic.Holistic() as holistic:
                            results = holistic.process(img_rgb)
                            
                            landmarks = {}
                            
                            if results.pose_landmarks:
                                landmarks['pose'] = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
                            if results.face_landmarks:
                                landmarks['face'] = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
                            if results.left_hand_landmarks:
                                landmarks['left_hand'] = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                            if results.right_hand_landmarks:
                                landmarks['right_hand'] = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                            
                            data_aux.append(landmarks)
                            data.append(data_aux)
                            labels.append(class_dir)


with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data and labels have been saved to pickle file")

with open('data.txt', 'w') as f:
    for i in range(len(data)):
        f.write(f"Label: {labels[i]}\n")
        f.write(f"Data: {data[i]}\n\n")

print("Data and labels have been saved to text file")


with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Label', 'Data'])
    for i in range(len(data)):
        writer.writerow([labels[i], data[i]])

print("Data and labels have been saved to CSV file")
