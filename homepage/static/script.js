        let inputMethod = "";
        const fileInput = document.getElementById('file-input');
        const videoInput = document.getElementById('video-input');
        const youtubeInput = document.getElementById('youtube-input');
        const captureFrame = document.getElementById('capture-frame');
        const displayImage = document.getElementById('display-image');
        const displayVideo = document.getElementById('display-video');
        const displayYouTube = document.getElementById('display-youtube');
        const webcamFeed = document.getElementById('webcam-feed');
        const controlButtons = document.getElementById('control-buttons');

        function handleInputChange() {
            inputMethod = document.getElementById('input-method').value;
            fileInput.style.display = 'none';
            videoInput.style.display = 'none';
            youtubeInput.style.display = 'none';
            displayImage.style.display = 'none';
            displayVideo.style.display = 'none';
            displayYouTube.style.display = 'none';
            webcamFeed.style.display = 'none';
            controlButtons.style.display = 'none';

            if (inputMethod === 'image') {
                fileInput.style.display = 'block';
            } else if (inputMethod === 'video') {
                videoInput.style.display = 'block';
            } else if (inputMethod === 'youtube') {
                youtubeInput.style.display = 'block';
                controlButtons.style.display = 'block';
            } else if (inputMethod === 'webcam') {
                webcamFeed.style.display = 'block';
                startWebcam();
            }
        }

        function loadFile(event) {
            const file = event.target.files[0];
            const formData = new FormData();
            formData.append(inputMethod, file);

            fetch(`/predict_${inputMethod}/`, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': '{{ csrf_token }}'
                }
            })
            .then(response => response.json())
            .then(data => {
                displayPredictions(data.predicted_labels, data.predicted_probabilities);
            });
        }

        function loadYouTubeVideo() {
            const url = youtubeInput.value;
            const videoId = new URL(url).searchParams.get('v');
            displayYouTube.src = `https://www.youtube.com/embed/${videoId}`;
            displayYouTube.style.display = 'block';

            // Add event listeners for keyboard controls
            document.addEventListener('keydown', handleKeyControls);
        }

        function handleKeyControls(event) {
            const iframe = displayYouTube;
            const iframeWindow = iframe.contentWindow;
            
            if (event.key === '>') {
                iframeWindow.postMessage('{"event":"command","func":"seekTo","args":[10, true]}', '*');
            } else if (event.key === '<') {
                iframeWindow.postMessage('{"event":"command","func":"seekTo","args":[-10, true]}', '*');
            } else if (event.key === ' ') {
                iframeWindow.postMessage('{"event":"command","func":"pauseVideo","args":""}', '*');
            }
        }

        function startCapture() {
            const url = youtubeInput.value;
            fetch(`/predict_youtube/`, {
                method: 'POST',
                body: JSON.stringify({ url: url, action: 'start_capture' }),
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': '{{ csrf_token }}'
                }
            })
            .then(response => response.json())
            .then(data => {
                displayPredictions(data.predicted_labels, data.predicted_probabilities);
            });
        }

        function stopCapture() {
            const url = youtubeInput.value;
            fetch(`/predict_youtube/`, {
                method: 'POST',
                body: JSON.stringify({ url: url, action: 'stop_capture' }),
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': '{{ csrf_token }}'
                }
            })
            .then(response => response.json())
            .then(data => {
                displayPredictions(data.predicted_labels, data.predicted_probabilities);
            });
        }

        function startWebcam() {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    webcamFeed.srcObject = stream;
                    webcamFeed.play();

                    const hands = new Hands({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
                    hands.setOptions({
                        maxNumHands: 1,
                        minDetectionConfidence: 0.5,
                        minTrackingConfidence: 0.5
                    });

                    hands.onResults(results => {
                        drawResults(results);
                        predictWebcamFrame();
                    });

                    async function predictWebcamFrame() {
                        const model = await tf.loadLayersModel('/static/model/model_alpha_1.json');
                        const labels = await fetch('/static/data/data_alpha_1.csv')
                            .then(response => response.text())
                            .then(text => text.split('\n').map(line => line.split(',')[0]));

                        async function framePrediction() {
                            hands.send({ image: webcamFeed });
                            const predictions = await predictFrame(webcamFeed, model, labels, hands);
                            displayPredictions(predictions.predicted_labels, predictions.predicted_probabilities);
                            requestAnimationFrame(framePrediction);
                        }

                        framePrediction();
                    }

                    function drawResults(results) {
                        const canvas = document.getElementById('landmarks-canvas');
                        canvas.width = webcamFeed.videoWidth;
                        canvas.height = webcamFeed.videoHeight;
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        if (results.multiHandLandmarks) {
                            for (const landmarks of results.multiHandLandmarks) {
                                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
                                drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 2 });
                            }
                        }
                    }
                });
        }

        function displayPredictions(predicted_labels, predicted_probabilities) {
            const predictionsElement = document.getElementById('predictions');
            predictionsElement.innerHTML = '';
            predicted_labels.forEach((label, index) => {
                const probability = predicted_probabilities[index].toFixed(2);
                const labelElement = document.createElement('div');
                labelElement.textContent = `${label}: ${probability}%`;
                predictionsElement.appendChild(labelElement);
            });

            const resultElement = document.getElementById('result');
            resultElement.textContent = predicted_labels[0];
        }

        function speakResult() {
            const result = document.getElementById('result').textContent;
            const utterance = new SpeechSynthesisUtterance(result);
            window.speechSynthesis.speak(utterance);
        }