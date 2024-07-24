import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pyttsx3
import threading
import queue
import mediapipe as mp
import time
from sklearn.preprocessing import StandardScaler
from pytubefix import YouTube
import tempfile
import os

engine = pyttsx3.init()
engine.setProperty('rate', 150)  
engine.setProperty('volume', 1)  

tts_queue = queue.Queue()

def load_labels(csv_file):
    df = pd.read_csv(csv_file, header=None)
    labels = df.iloc[:, 0].unique().tolist()
    return labels

def normalize(vector_axis):
    normalized = []
    axrange = max(vector_axis) - min(vector_axis)
    for value in vector_axis:
        normalized.append((value - min(vector_axis)) / axrange)
    return normalized

def preprocess_image(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            vector_x, vector_y, vector_z = [], [], []
            for landmark in hand_landmarks.landmark:
                vector_x.append(landmark.x)
                vector_y.append(landmark.y)
                vector_z.append(landmark.z)
            
            normalized_vector_x = normalize(vector_x)
            normalized_vector_y = normalize(vector_y)
            normalized_vector_z = normalize(vector_z)
            
            vector = np.concatenate([normalized_vector_x, normalized_vector_y, normalized_vector_z])
            return vector
    return None

def speak_text():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

def process_image(image_path, model, labels, hands):
    image = cv2.imread(image_path)
    vector = preprocess_image(image, hands)
    if vector is not None:
        vector = vector.reshape(1, -1)  
        predictions = model.predict(vector)
        predicted_probabilities = predictions[0]
        top_indices = np.argsort(predicted_probabilities)[-3:]  
        
        predicted_labels = [labels[i] for i in top_indices[::-1]]  
        predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]
        
        for i in range(3):
            cv2.putText(image, f"{predicted_labels[i]}: {predicted_probabilities[i]:.2f}%", (50, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Image Processing', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(video_path, model, labels, hands):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        vector = preprocess_image(frame, hands)
        if vector is not None:
            vector = vector.reshape(1, -1)  
            predictions = model.predict(vector)
            predicted_probabilities = predictions[0]
            top_indices = np.argsort(predicted_probabilities)[-3:]  
            
            predicted_labels = [labels[i] for i in top_indices[::-1]]  
            predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]

            for i in range(3):
                cv2.putText(frame, f"{predicted_labels[i]}: {predicted_probabilities[i]:.2f}%", (50, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_youtube(url, model, labels, hands):
    try:
        yt = YouTube(url)
        video_stream = yt.streams.get_highest_resolution()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
            video_path = temp_file.name

        process_video(video_path, model, labels, hands)
        
        os.remove(video_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def process_webcam(model, labels, hands):
    cap = cv2.VideoCapture(0)
    
    tts_thread = threading.Thread(target=speak_text)
    tts_thread.start()
    
    last_speech_time = time.time()
    speech_interval = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        vector = preprocess_image(frame, hands)
        if vector is not None:
            vector = vector.reshape(1, -1)  
            predictions = model.predict(vector)
            predicted_probabilities = predictions[0]
            top_indices = np.argsort(predicted_probabilities)[-3:]  
            
            predicted_labels = [labels[i] for i in top_indices[::-1]]  
            predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]

            for i in range(3):
                cv2.putText(frame, f"{predicted_labels[i]}: {predicted_probabilities[i]:.2f}%", (50, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            current_time = time.time()
            if current_time - last_speech_time > speech_interval:
                tts_queue.put(f"{predicted_labels[0]}")
                last_speech_time = current_time

        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    tts_queue.put(None)
    tts_thread.join()

def main():
    model = load_model('model_1.h5')
    labels = load_labels('data1.csv')
    hands = mp.solutions.hands.Hands()  
    
    while True:
        choice = input("Choose mode: (1) Image (2) Video (3) Webcam (4) YouTube (5) Exit: ")
        
        if choice == '1':
            image_path = input("Enter path to image file: ")
            process_image(image_path, model, labels, hands)
        elif choice == '2':
            video_path = input("Enter path to video file: ")
            process_video(video_path, model, labels, hands)
        elif choice == '3':
            process_webcam(model, labels, hands)
        elif choice == '4':
            youtube_url = input("Enter YouTube video URL: ")
            process_youtube(youtube_url, model, labels, hands)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
