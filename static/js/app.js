const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const clearBtn = document.getElementById('clear-btn');
const speakBtn = document.getElementById('speak-btn');
const themeToggle = document.getElementById('theme-toggle');
const handIndicator = document.getElementById('hand-indicator');
const gestureDisplay = document.getElementById('current-gesture');
const sentenceBox = document.getElementById('sentence-box');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const switchCameraBtn = document.getElementById('switch-camera-btn');
const videoWrapper = document.getElementById('video-wrapper');

let currentSentence = "";
let camera = null;
let currentFacingMode = 'user'; // 'user' for front, 'environment' for back
let lastEmitTime = 0;
const EMIT_INTERVAL = 200; // Send landmark every 200ms (5 times per second)
const socket = io();

// Stability tracking
let lastPredictedGesture = "";
let gestureStartTime = 0;
const STABILITY_TIME = 700; // Reduced from 1000ms to 700ms for faster feel

// Theme Toggle
themeToggle.addEventListener('click', () => {
    const body = document.body;
    const icon = themeToggle.querySelector('i');
    if (body.getAttribute('data-theme') === 'dark') {
        body.removeAttribute('data-theme');
        icon.className = 'fas fa-moon';
    } else {
        body.setAttribute('data-theme', 'dark');
        icon.className = 'fas fa-sun';
    }
});

// Fullscreen Logic
fullscreenBtn.addEventListener('click', () => {
    if (!document.fullscreenElement) {
        if (videoWrapper.requestFullscreen) {
            videoWrapper.requestFullscreen();
        } else if (videoWrapper.webkitRequestFullscreen) { /* Safari */
            videoWrapper.webkitRequestFullscreen();
        } else if (videoWrapper.msRequestFullscreen) { /* IE11 */
            videoWrapper.msRequestFullscreen();
        }
        fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) { /* Safari */
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) { /* IE11 */
            document.msExitFullscreen();
        }
        fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
    }
});

// Update fullscreen button icon on escape
document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
        fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
    }
});

// MediaPipe Hands Setup
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

function onResults(results) {
    // Resize canvas to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw landmarks
    if (results.multiHandLandmarks) {
        const numHands = results.multiHandLandmarks.length;
        if (numHands > 0) {
            handIndicator.textContent = `${numHands} Hand${numHands > 1 ? 's' : ''} Detected`;
            handIndicator.classList.add('active');
            
            // Process the first hand for prediction
            const landmarks = results.multiHandLandmarks[0];
            const flattenedLandmarks = [];
            landmarks.forEach(lm => {
                flattenedLandmarks.push(lm.x, lm.y, lm.z);
            });

            // Throttled emission: only send if interval has passed
            const now = Date.now();
            if (now - lastEmitTime > EMIT_INTERVAL) {
                socket.emit('predict', { landmarks: flattenedLandmarks });
                lastEmitTime = now;
            }
        } else {
            handIndicator.textContent = "No Hand Detected";
            handIndicator.classList.remove('active');
            gestureDisplay.textContent = "-";
        }

        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#6366f1', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#ffffff', lineWidth: 2, radius: 4});
        }
    }
    canvasCtx.restore();
}

// Socket prediction result
socket.on('prediction_result', (data) => {
    if (data.error) {
        console.error(data.error);
        return;
    }
    
    const detectedGesture = data.gesture;
    gestureDisplay.textContent = detectedGesture;
    
    // Stability logic
    if (detectedGesture === "-" || detectedGesture === "NONE") {
        lastPredictedGesture = "";
        gestureStartTime = 0;
        return;
    }

    if (detectedGesture !== lastPredictedGesture) {
        lastPredictedGesture = detectedGesture;
        gestureStartTime = Date.now();
    } else {
        if (gestureStartTime > 0) {
            const elapsedTime = Date.now() - gestureStartTime;
            if (elapsedTime >= STABILITY_TIME) {
                if (currentSentence === "Waiting for input...") currentSentence = "";
                
                currentSentence += detectedGesture;
                sentenceBox.textContent = currentSentence;
                
                const utterance = new SpeechSynthesisUtterance(detectedGesture);
                window.speechSynthesis.speak(utterance);

                gestureStartTime = 0; 
                lastPredictedGesture = ""; 
            }
        }
    }
});

async function startCamera(facingMode = 'user') {
    if (camera) {
        await camera.stop();
    }

    camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({image: videoElement});
        },
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
    });
    
    await camera.start();
    currentFacingMode = facingMode;
}

// Camera Controls
startBtn.addEventListener('click', async () => {
    await startCamera(currentFacingMode);
    startBtn.disabled = true;
    stopBtn.disabled = false;
});

stopBtn.addEventListener('click', () => {
    if (camera) {
        camera.stop();
        const stream = videoElement.srcObject;
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
        }
        videoElement.srcObject = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    handIndicator.textContent = "Camera Stopped";
    handIndicator.classList.remove('active');
});

switchCameraBtn.addEventListener('click', async () => {
    const newMode = currentFacingMode === 'user' ? 'environment' : 'user';
    await startCamera(newMode);
});

clearBtn.addEventListener('click', () => {
    currentSentence = "";
    sentenceBox.textContent = "Waiting for input...";
    gestureDisplay.textContent = "-";
});

speakBtn.addEventListener('click', () => {
    const text = sentenceBox.textContent;
    if (text && text !== "Waiting for input...") {
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
});
