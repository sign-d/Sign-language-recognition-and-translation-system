#minus standardization

import cv2
import mediapipe as mp
import pandas as pd
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

def normalize(vector_axis):
    normalized = []
    axrange = max(vector_axis) - min(vector_axis)
    for value in vector_axis:
        normalized.append((value - min(vector_axis)) / axrange)
    return normalized

def process_image(image_path, hands):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            vector, vector_x, vector_y, vector_z = [], [], [], []

            for landmark in hand_landmarks.landmark:
                vector_x.append(landmark.x)
                vector_y.append(landmark.y)
                vector_z.append(landmark.z)
            
            vector.extend(normalize(vector_x))
            vector.extend(normalize(vector_y))
            vector.extend(normalize(vector_z))
            
            return vector
    return None

def process_directory(directory_path, hands):
    data_list = []
    for category in os.listdir(directory_path):
        category_path = os.path.join(directory_path, category)
        if os.path.isdir(category_path):
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                vector = process_image(image_path, hands)
                if vector is not None:
                    vector.insert(0, category)
                    data_list.append(vector)
    return data_list

def main():
    main_directory = "data"
    hands = mphands.Hands()
    data_list = []

    for data_type in os.listdir(main_directory):
        data_type_path = os.path.join(main_directory, data_type)
        if os.path.isdir(data_type_path):
            data_list.extend(process_directory(data_type_path, hands))
    
    df = pd.DataFrame(data_list)
    
    df.to_csv("data2.csv", index=False, header=False)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
