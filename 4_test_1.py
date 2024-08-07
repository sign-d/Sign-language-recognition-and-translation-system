import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import tempfile
import os
from pytubefix import YouTube
import mediapipe as mp
import time
import re
from fuzzywuzzy import process

def load_labels(csv_file):
    try:
        df = pd.read_csv(csv_file, header=None)
        labels = df.iloc[:, 0].unique().tolist()
        return labels
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

def normalize(vector_axis):
    vector_axis = np.array(vector_axis)
    axrange = vector_axis.max() - vector_axis.min()
    return (vector_axis - vector_axis.min()) / axrange

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

def predict(vector, model, labels):
    vector = vector.reshape(1, -1)
    predictions = model.predict(vector)
    predicted_probabilities = predictions[0]
    top_indices = np.argsort(predicted_probabilities)[-5:]

    predicted_labels = [labels[i] for i in top_indices[::-1]]
    predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]
    
    # Filter predictions based on probability range
    filtered_labels = []
    filtered_probabilities = []
    for label, prob in zip(predicted_labels, predicted_probabilities):
        if 98 <= prob <= 100:
            filtered_labels.append(label)
            filtered_probabilities.append(prob)
    
    return filtered_labels, filtered_probabilities

def display_predictions(image, labels, probabilities):
    for i in range(len(labels)):  # Adjusted to handle the exact number of labels predicted
        cv2.putText(image, f"{labels[i]}: {probabilities[i]:.2f}%", (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def process_image(image_path, model, labels, hands, results_dict):
    image = cv2.imread(image_path)
    vector = preprocess_image(image, hands)
    if vector is not None:
        predicted_labels, predicted_probabilities = predict(vector, model, labels)
        if predicted_labels:
            highest_prob_index = np.argmax(predicted_probabilities)
            highest_prediction = predicted_labels[highest_prob_index]
            
            # Store the result in the dictionary
            results_dict[image_path] = highest_prediction
            
            display_predictions(image, predicted_labels, predicted_probabilities)
            print(f"Highest prediction for image {image_path}: {highest_prediction}")
            cv2.imshow('Image Processing', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No predictions with high enough confidence.")
    else:
        print("No hand landmarks detected in the image.")

def process_video(video_path, model, labels, hands, skip_time, results_dict):
    cap = cv2.VideoCapture(video_path)
    
    # Skip the initial frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(skip_time * frame_rate)
    for _ in range(skip_frames):
        ret, frame = cap.read()
        if not ret:
            break

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vector = preprocess_image(frame, hands)
        if vector is not None:
            predicted_labels, predicted_probabilities = predict(vector, model, labels)
            if predicted_labels:
                highest_prob_index = np.argmax(predicted_probabilities)
                highest_prediction = predicted_labels[highest_prob_index]
                
                # Store the result with timestamp in the dictionary
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # Get timestamp in seconds
                results_dict[timestamp] = highest_prediction
                
                display_predictions(frame, predicted_labels, predicted_probabilities)
                print(f"Value predicted {highest_prediction} at timestamp {timestamp}")
                
        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Process and display the final results dictionary
    concatenated_text = ''.join([value for value in results_dict.values()])
    words = find_words(concatenated_text)
    sentence = form_sentence(words)
    print(f"Concatenated detected letters: {concatenated_text}")
    print(f"Formed sentence: {sentence}")
    

def process_youtube(url, model, labels, hands, skip_time, results_dict):
    try:
        yt = YouTube(url)
        video_stream = yt.streams.get_highest_resolution()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
            video_path = temp_file.name

        process_video(video_path, model, labels, hands, skip_time, results_dict)
        
        os.remove(video_path)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

def process_webcam(model, labels, hands, results_dict):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vector = preprocess_image(frame, hands)
        if vector is not None:
            predicted_labels, predicted_probabilities = predict(vector, model, labels)
            if predicted_labels:
                highest_prob_index = np.argmax(predicted_probabilities)
                highest_prediction = predicted_labels[highest_prob_index]

                # Store the result with timestamp in the dictionary
                timestamp = int(time.time())
                results_dict[timestamp] = highest_prediction

                display_predictions(frame, predicted_labels, predicted_probabilities)
                print(f"Highest prediction for webcam: {highest_prediction}")

        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_dictionary(filename):
    with open(filename) as f:
        return set(word.strip().lower() for word in f)

dictionary = load_dictionary('dictionary.txt')

def fuzzy_match(word, dictionary, threshold=20):
    if not word:
        return []
    matches = process.extractBests(word, dictionary, score_cutoff=threshold)
    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return sorted_matches

def fuzzy_match_best(word, dictionary, threshold=90):
    if not word:
        return None
    best_match = process.extractOne(word, dictionary, score_cutoff=threshold)
    print(best_match)
    return best_match

def find_words(input_text):
    words = []
    length = len(input_text)
    start = 0

    while start < length:
        best_match = ("", 0)  # (word, score)
        end = start + 1

        while end <= length:
            segment = input_text[start:end]
            matches = fuzzy_match(segment, dictionary)
            if matches:
                for match in matches:
                    word, score = match
                    if score > best_match[1]:
                        best_match = (word, score)

            end += 1

        if best_match[0]:
            words.append(best_match[0])
            start += len(best_match[0])
        else:
            start += 1  # Move to the next character if no match found

    return words

def form_sentence(words):
    return ' '.join(words)

def main():
    model = load_model('model_alpha_1.h5')
    labels = load_labels('data_alpha_1.csv')
    hands = mp.solutions.hands.Hands()
    results_dict = {}  
    
    while True:
        choice = input("Choose mode: (1) Image (2) Video (3) YouTube (4) Webcam (5) Quit: ")
        
        if choice == '1':
            image_path = input("Enter image file path: ")
            process_image(image_path, model, labels, hands, results_dict)
        elif choice == '2':
            video_path = input("Enter video file path: ")
            skip_time = float(input("Enter time to skip in seconds: "))
            process_video(video_path, model, labels, hands, skip_time, results_dict)
        elif choice == '3':
            url = input("Enter YouTube video URL: ")
            skip_time = float(input("Enter time to skip in seconds: "))
            process_youtube(url, model, labels, hands, skip_time, results_dict)
        elif choice == '4':
            process_webcam(model, labels, hands, results_dict)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

    print("Processing completed.")
    concatenated_text = ''.join([value for value in results_dict.values()])
    words = find_words(concatenated_text)
    sentence = form_sentence(words)
    print(f"Concatenated detected letters: {concatenated_text}")
    print(f"Formed sentence: {sentence}")
    
if __name__ == "__main__":
    main()
