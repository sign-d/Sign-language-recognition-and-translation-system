import os
import cv2
from pytube import YouTube


def get_youtube_stream_url(video_url):
    yt = YouTube(video_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    return stream.url


stream_url = get_youtube_stream_url('https://www.youtube.com/watch?v=gEoS1e764u0')


root_path = 'data'
classes = {}
files = 30
video_path = stream_url


cap = cv2.VideoCapture('video.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not os.path.exists(root_path):
    os.makedirs(root_path)

while True:
    class_name = input('Enter class name: ')
    if class_name == 'q':
        break

    classes[class_name] = len(classes)

    class_path = os.path.join(root_path, class_name)

    if not os.path.exists(class_path):
        os.makedirs(class_path)
        print(f'Folder {class_name} created')

        counter = 0
    else:
        print(f'Folder {class_name} already exists')
        existing_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        if len(existing_files) == files:
            print(f'Folder {class_name} is full')
            continue
        if existing_files:
            existing_files.sort(key=lambda f: int(os.path.splitext(f)[0]))
            last_file = existing_files[-1]
            counter = int(os.path.splitext(last_file)[0]) + 1
        else:
            counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from video. Exiting.")
            break

        text_1 = 'Press q to start collecting data collecting data for ' + class_name
        frame_height, frame_width = frame.shape[:2]
        (text_width, text_height), _ = cv2.getTextSize(text_1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (frame_width - text_width) // 2
        y = 50 // 2

        cv2.rectangle(frame, (0, 0), (1280, 40), (255, 0, 0), -1)
        cv2.putText(frame, text_1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == 27:
            print('You exited the camera')
            cap.release()
            cv2.destroyAllWindows()
            exit()

    paused = False
    rewind = False
    forward = False

    while counter < files:

        if not paused:
            if rewind:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 10)
            elif forward:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 10)

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from video. Exiting.")
            break

        text_2 = 'Collecting data for ' + class_name + ' (' + str(counter) + '/' + str(files) + ')'
        cv2.rectangle(frame, (0, 0), (1280, 40), (255, 0, 0), -1)
        cv2.putText(frame, f"{text_2}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
            print(f'Frame {counter} saved')
            counter += 1
            if counter == files:
                break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            rewind = True
            forward = False
        elif key == ord('f'):
            forward = True
            rewind = False
        elif key == ord('s'):
            break
        elif key == ord('c'):
            rewind = False
            forward = False
        elif key == 27:
            print('You exited the camera')
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
