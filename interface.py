import sys
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import pyttsx3
import threading
import queue
import mediapipe as mp
import time
import tempfile
import os
from pytubefix import YouTube
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFileDialog, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# Initialize TTS Engine and Queue
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1)  # Set volume level
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
            return vector, hand_landmarks  # Return both the vector and landmarks
    return None, None

def speak_text():
    while True:
        texts = tts_queue.get()
        if texts is None:
            break
        for text in texts:
            time.sleep(2)  # Delay before speaking
            engine.say(text)
            engine.runAndWait()

class SignLanguageRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = tf.keras.models.load_model('model_alpha_1.h5')
        self.labels = load_labels('data_alpha_1.csv')
        self.hands = mp.solutions.hands.Hands()
        self.current_mode = None
        self.file_path = None
        self.youtube_url = None
        self.tts_thread = threading.Thread(target=speak_text)
        self.tts_thread.start()
        self.last_speech_time = time.time()  # Track the last speech time

    def initUI(self):
        self.setWindowTitle('Sign Language Recognizer')
        self.setGeometry(100, 100, 800, 600)  # Set default window size

        # Create labels for video feed and landmarks
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)  # Allow the video label to scale
        self.landmark_label = QLabel(self)
        self.landmark_label.setScaledContents(True)  # Allow the landmark label to scale
        self.text_label = QLabel(self)

        # Create buttons and combo box
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Select Mode", "Webcam", "Image", "Video", "YouTube"])
        self.file_button = QPushButton('Select File', self)
        self.url_button = QPushButton('Enter YouTube URL', self)

        # Set up layout
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        display_layout = QHBoxLayout()

        display_layout.addWidget(self.video_label)
        display_layout.addWidget(self.landmark_label)

        control_layout.addWidget(self.mode_combo)
        control_layout.addWidget(self.file_button)
        control_layout.addWidget(self.url_button)

        main_layout.addLayout(display_layout)
        main_layout.addWidget(self.text_label)
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

        # Connect signals and slots
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        self.file_button.clicked.connect(self.open_file_dialog)
        self.url_button.clicked.connect(self.open_url_dialog)
        self.file_button.setEnabled(False)
        self.url_button.setEnabled(False)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def load_and_compile_model(self):
        if not self.model.optimizer:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def update_frame(self):
        if self.current_mode == "Webcam":
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame_rgb, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), aspectRatioMode=1))

                # Process the frame and recognize the sign
                vector, hand_landmarks = preprocess_image(frame, self.hands)
                if vector is not None:
                    vector = vector.reshape(1, -1)
                    predictions = self.model.predict(vector)
                    predicted_probabilities = predictions[0]
                    top_index = np.argmax(predicted_probabilities)
                    sign_text = self.labels[top_index]
                    
                    # Display the hand landmarks
                    if hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    self.text_label.setText(sign_text)
                    tts_queue.put([sign_text])

                    # Update landmark image
                    landmark_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmark_qimage = QImage(landmark_image, landmark_image.shape[1], landmark_image.shape[0], QImage.Format_RGB888)
                    self.landmark_label.setPixmap(QPixmap.fromImage(landmark_qimage).scaled(self.landmark_label.size(), aspectRatioMode=1))

                current_time = time.time()
                if current_time - self.last_speech_time > 2:
                    self.last_speech_time = current_time

        elif self.current_mode == "Image":
            if self.file_path:
                image = cv2.imread(self.file_path)
                vector, _ = preprocess_image(image, self.hands)
                if vector is not None:
                    vector = vector.reshape(1, -1)
                    predictions = self.model.predict(vector)
                    predicted_probabilities = predictions[0]
                    top_indices = np.argsort(predicted_probabilities)[-5:]

                    predicted_labels = [self.labels[i] for i in top_indices[::-1]]
                    predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]

                    for i in range(5):
                        cv2.putText(image, f"{predicted_labels[i]}: {predicted_probabilities[i]:.2f}%", (50, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.imshow('Image Processing', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        elif self.current_mode == "Video":
            if self.file_path:
                self.process_video(self.file_path)
        
        elif self.current_mode == "YouTube":
            if self.youtube_url:
                self.process_youtube(self.youtube_url)
        
        # Check if 'q' is pressed to exit all windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def recognize_sign(self, frame):
        vector, _ = preprocess_image(frame, self.hands)
        if vector is not None:
            vector = vector.reshape(1, -1)
            predictions = self.model.predict(vector)
            predicted_probabilities = predictions[0]
            top_index = np.argmax(predicted_probabilities)
            return self.labels[top_index]
        return "Unknown"

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            vector, _ = preprocess_image(frame, self.hands)
            if vector is not None:
                vector = vector.reshape(1, -1)
                predictions = self.model.predict(vector)
                predicted_probabilities = predictions[0]
                top_indices = np.argsort(predicted_probabilities)[-5:]
                
                predicted_labels = [self.labels[i] for i in top_indices[::-1]]
                predicted_probabilities = [predicted_probabilities[i] * 100 for i in top_indices[::-1]]

                for i in range(5):
                    cv2.putText(frame, f"{predicted_labels[i]}: {predicted_probabilities[i]:.2f}%", (50, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Video Processing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def process_youtube(self, url):
        try:
            yt = YouTube(url)
            video_stream = yt.streams.get_highest_resolution()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                video_stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
                video_path = temp_file.name

            self.process_video(video_path)
            
            os.remove(video_path)
        except Exception as e:
            print(f"An error occurred: {e}")

    def change_mode(self):
        self.current_mode = self.mode_combo.currentText()
        if self.current_mode in ["Image", "Video", "YouTube"]:
            self.file_button.setEnabled(True)
            self.url_button.setEnabled(self.current_mode == "YouTube")
        else:
            self.file_button.setEnabled(False)
            self.url_button.setEnabled(False)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Image Files (*.jpg *.jpeg *.png);;Video Files (*.mp4)", options=options)
        if file_path:
            self.file_path = file_path
            self.update_frame()

    def open_url_dialog(self):
        url, _ = QInputDialog.getText(self, "Enter YouTube URL", "YouTube URL:")
        if url:
            self.youtube_url = url
            self.update_frame()

    def closeEvent(self, event):
        self.cap.release()
        tts_queue.put(None)
        self.tts_thread.join()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    recognizer = SignLanguageRecognizer()
    recognizer.show()
    sys.exit(app.exec_())
