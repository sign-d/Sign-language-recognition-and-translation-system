function handleInputChange() {
    const inputMethod = document.getElementById('input-method').value;
    const fileInput = document.getElementById('file-input');
    const videoInput = document.getElementById('video-input');
    const youtubeInput = document.getElementById('youtube-input');
    const inputWindow = document.getElementById('input-window');

    fileInput.style.display = 'none';
    videoInput.style.display = 'none';
    youtubeInput.style.display = 'none';

    if (inputMethod === 'image') {
        fileInput.style.display = 'block';
    } else if (inputMethod === 'video') {
        videoInput.style.display = 'block';
    } else if (inputMethod === 'webcam') {
        startWebcam();
    } else if (inputMethod === 'youtube') {
        youtubeInput.style.display = 'block';
    }
}

function loadFile(event) {
    const inputWindow = document.getElementById('input-window');
    const file = event.target.files[0];
    const url = URL.createObjectURL(file);

    if (file.type.startsWith('image/')) {
        const img = document.createElement('img');
        img.src = url;
        img.style.width = '100%';
        img.style.height = '100%';
        inputWindow.innerHTML = '';
        inputWindow.appendChild(img);
    } else if (file.type.startsWith('video/')) {
        const video = document.createElement('video');
        video.src = url;
        video.style.width = '100%';
        video.style.height = '100%';
        video.controls = true;
        inputWindow.innerHTML = '';
        inputWindow.appendChild(video);
    }
}

function startWebcam() {
    const inputWindow = document.getElementById('input-window');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            inputWindow.innerHTML = '';
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();
            inputWindow.appendChild(video);
        })
        .catch(err => console.error('Error accessing webcam:', err));
}

function speakResult() {
    const resultText = document.getElementById('result').textContent;
    const speech = new SpeechSynthesisUtterance(resultText);
    window.speechSynthesis.speak(speech);
}

function fetchPredictions(inputData) {
    // Dummy function to simulate fetching predictions
    // Replace with actual logic to call your ML model and get predictions
    const predictions = [
        { label: 'Sign A', confidence: 0.7 },
        { label: 'Sign B', confidence: 0.3 },
    ];

    displayPredictions(predictions);
}

function displayPredictions(predictions) {
    const predictionsOutput = document.getElementById('predictions');
    predictionsOutput.innerHTML = predictions.map(p => `${p.label}: ${(p.confidence * 100).toFixed(2)}%`).join('<br>');

    const highestPrediction = predictions.reduce((max, p) => p.confidence > max.confidence ? p : max, predictions[0]);
    document.getElementById('result').textContent = highestPrediction.label;
}

window.onbeforeunload = function () {
    document.getElementById('input-window').innerHTML = '';
    document.getElementById('landmarks-window').innerHTML = '';
    document.getElementById('predictions').innerHTML = '';
    document.getElementById('result').innerHTML = '';
};
