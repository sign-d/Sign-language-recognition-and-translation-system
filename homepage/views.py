import io
from django.shortcuts import render
from django.contrib.auth import login, authenticate
import requests
from urllib3 import Retry
from .forms import SignUpForm, LoginForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import numpy as np
import tensorflow as tf
import tempfile
import pandas as pd
import cv2
import pyttsx3
import mediapipe as mp
import yt_dlp as youtube_dl
import queue
from django.conf import settings
import threading
import time
import os
import logging
from tensorflow.keras.models import load_model
from collections import defaultdict
from pytube import YouTube
import ffmpeg
import re
import streamlink
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'homepage/index.html')

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'error', 'errors': ['Invalid login credentials']})
        else:
            errors = [error for error in form.errors.values()]
            return JsonResponse({'status': 'error', 'errors': errors})
    else:
        form = LoginForm()
    return render(request, 'homepage/login.html', {'form': form})

@csrf_exempt
def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({'status': 'success'})
        else:
            errors = [error for error in form.errors.values()]
            return JsonResponse({'status': 'error', 'errors': errors})
    else:
        form = SignUpForm()
    return render(request, 'homepage/signup.html', {'form': form})

def about_view(request):
    return render(request, 'homepage/about.html')

def recog_view(request):
    return render(request, 'homepage/recog.html')

engine = pyttsx3.init()
engine.setProperty('rate', 150)  
engine.setProperty('volume', 1)  

tts_queue = queue.Queue()

def compile_model(model):
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

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
        texts = tts_queue.get()
        if texts is None:
            break
        for text in texts:
            engine.say(text)
            engine.runAndWait()

def predict_frame(frame, model, labels, hands):
    vector = preprocess_image(frame, hands)
    if vector is not None:
        vector = vector.reshape(1, -1)  
        predictions = model.predict(vector)
        predicted_probabilities = predictions[0]
        top_indices = np.argsort(predicted_probabilities)[-5:]
        
        predicted_labels = [labels[i] for i in top_indices[::-1]]
        predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]
        
        return predicted_labels, predicted_probabilities
    return None, None

@csrf_exempt
def webcam_feed(request):
    cap = cv2.VideoCapture(0)
    model = load_model('homepage/model_alpha_1.h5')
    compile_model(model)
    labels = load_labels('homepage/data_alpha_1.csv')
    hands = mp.solutions.hands.Hands()

    last_speech_time = time.time()
    speech_interval = 2  # 2 seconds interval

    tts_thread = threading.Thread(target=speak_text)
    tts_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_labels, predicted_probabilities = predict_frame(frame, model, labels, hands)

        if predicted_labels and predicted_probabilities:
            current_time = time.time()
            if current_time - last_speech_time > speech_interval:
                tts_queue.put([predicted_labels[0]])
                last_speech_time = current_time
            logger.info('Model successfully recognized the input frame from webcam.')
            logger.info('Predicted labels: %s', predicted_labels)
            logger.info('Predicted probabilities: %s', predicted_probabilities)            
            
            response_data = {
                'predicted_labels': predicted_labels,
                'predicted_probabilities': predicted_probabilities
            }
            cap.release()
            cv2.destroyAllWindows()
            return JsonResponse(response_data)

    tts_queue.put(None)
    tts_thread.join()
    return JsonResponse({'status': 'stopped'})

@csrf_exempt
def predict_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image = request.FILES['image']
            path = default_storage.save('temp_image.jpg', image)
            img = cv2.imread(path)

            model = load_model('homepage/model_alpha_1.h5')
            compile_model(model)
            labels = load_labels('homepage/data_alpha_1.csv')
            hands = mp.solutions.hands.Hands()

            predicted_labels, predicted_probabilities = predict_frame(img, model, labels, hands)

            logger.info('Model successfully recognized the input image.')
            logger.info('Predicted labels: %s', predicted_labels)
            logger.info('Predicted probabilities: %s', predicted_probabilities)

            os.remove(path)  # Cleanup

            return JsonResponse({
                'predicted_labels': predicted_labels,
                'predicted_probabilities': predicted_probabilities
            })

        except Exception as e:
            logger.error('Error in predicting image: %s', str(e))
            return JsonResponse({'status': 'error', 'message': 'Prediction failed'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@csrf_exempt
def predict_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            video = request.FILES['video']
            path = default_storage.save('temp_video.mp4', video)
            cap = cv2.VideoCapture(path)

            model = load_model('homepage/model_alpha_1.h5')
            compile_model(model)
            labels = load_labels('homepage/data_alpha_1.csv')
            hands = mp.solutions.hands.Hands()

            predictions = []
            all_probabilities = {label: [] for label in labels}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                predicted_labels, predicted_probabilities = predict_frame(frame, model, labels, hands)
                if predicted_labels and predicted_probabilities:
                    predictions.append((predicted_labels, predicted_probabilities))
                    for label, prob in zip(predicted_labels, predicted_probabilities):
                        all_probabilities[label].append(prob)

            cap.release()
            os.remove(path)  # Cleanup

            average_probabilities = {label: np.mean(probs) for label, probs in all_probabilities.items()}
            top_predictions = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)
            top_predictions = top_predictions[:5]
            top_predicted_labels = [label for label, _ in top_predictions]
            top_predicted_probabilities = [prob for _, prob in top_predictions]

            logger.info('Model successfully recognized the input video.')
            logger.info('Predicted labels: %s', predictions)
            logger.info('Top predictions: %s', top_predictions)

            return JsonResponse({
                'predictions': predictions,
                'predicted_labels': top_predicted_labels,
                'predicted_probabilities': top_predicted_probabilities,
                'frame_count': len(predictions)
            })
        
        except Exception as e:
            logger.error('Error in predicting video: %s', str(e))
            return JsonResponse({'status': 'error', 'message': 'Prediction failed'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})



capture_running = False
# Create a session with retry logic
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Update the user agent to avoid detection as a bot
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}

def download_video(url, start_time=None, end_time=None):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'restrictfilenames': True,
        'nocheckcertificate': True,
        'headers': HEADERS,
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            video_filename = ydl.prepare_filename(result)
            
            if start_time is not None and end_time is not None:
                video_clip = mp.VideoFileClip(video_filename).subclip(start_time, end_time)
                output_filename = os.path.join(temp_dir, f"{result['id']}_trimmed.mp4")
                video_clip.write_videofile(output_filename, codec='libx264')
                video_filename = output_filename

        return video_filename

    except youtube_dl.DownloadError as e:
        print(f"Error downloading video: {e}")
        return None

def process_youtube_video(video_url, start_time=None, end_time=None):
    video_filename = download_video(video_url, start_time, end_time)
    
    if video_filename:
        print(f"Video downloaded successfully: {video_filename}")
        # Process the video (e.g., run prediction model)
        # ...
        # Cleanup temporary directory
        shutil.rmtree(Path(video_filename).parent)
    else:
        print("Failed to download video.")

# Example usage
video_url = 'https://www.youtube.com/watch?v=LHcaGxQDzDM'
start_time = 10  # Start time in seconds
end_time = 20    # End time in seconds

process_youtube_video(video_url, start_time, end_time)