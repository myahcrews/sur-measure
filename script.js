import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm";

// ========================================
// PAGE ELEMENTS
// ========================================

const photoUpload = document.getElementById("photo-upload");
const previewImage = document.getElementById("preview-image");
const placeholderText = document.getElementById("placeholder-text");

const landmarkCanvas = document.getElementById("landmark-canvas");
const canvasContext = landmarkCanvas.getContext("2d");

const measurementsSection =
    document.getElementById("measurements-section");

let currentImageURL = null;
let faceLandmarker = null;
let uploadVersion = 0;

// Keep landmark choices together so they are easy to update.
const POINTS = {
    cheekA: 234,
    cheekB: 454,

    foreheadA: 54,
    foreheadB: 284,

    foreheadTop: 10,
    chin: 152,

    jawA: 172,
    jawB: 397,

    rightEyeOuter: 33,
    rightEyeInner: 133,
    rightEyeUpper: 159,
    rightEyeLower: 145,

    leftEyeOuter: 263,
    leftEyeInner: 362,
    leftEyeUpper: 386,
    leftEyeLower: 374
};

// ========================================
// LOAD FACE LANDMARK MODEL
// ========================================

async function createFaceLandmarker() {
    console.log("Loading Face Landmarker...");

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
    );

    const options = {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",

            delegate: "GPU"
        },

        runningMode: "IMAGE",
        numFaces: 1
    };

    try {
        faceLandmarker = await FaceLandmarker.createFromOptions(
            vision,
            options
        );
    } catch (error) {
        console.warn("GPU unavailable. Trying CPU.", error);

        options.baseOptions.delegate = "CPU";

        faceLandmarker = await FaceLandmarker.createFromOptions(
            vision,
            options
        );
    }

    console.log("Face Landmarker loaded.");
}

const modelReady = createFaceLandmarker();

// Handle loading failures even before someone uploads a photo.
modelReady.catch((error) => {
    console.error("Face model could not load:", error);

    placeholderText.textContent =
        "The face model could not load. Refresh the page to try again.";
});

// ========================================
// HELPERS
// ========================================

function displayValue(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}

function toPixels(landmark) {
    return {
        x: landmark.x * previewImage.naturalWidth,
        y: landmark.y * previewImage.naturalHeight
    };
}

function getLandmarkDistance(pointA, pointB) {
    const a = toPixels(pointA);
    const b = toPixels(pointB);

    return Math.hypot(b.x - a.x, b.y - a.y);
}

// ========================================
// PHOTO UPLOAD
// ========================================

photoUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (!file) {
        return;
    }

    const thisUpload = ++uploadVersion;

    measurementsSection.hidden = true;
    landmarkCanvas.style.display = "none";
    previewImage.style.display = "none";

    placeholderText.textContent = "Loading your photo...";
    placeholderText.style.display = "block";

    if (currentImageURL) {
        URL.revokeObjectURL(currentImageURL);
    }

    currentImageURL = URL.createObjectURL(file);

    // Register handlers before assigning the image source.
    previewImage.onload = async function () {
        if (thisUpload !== uploadVersion) {
            return;
        }

        previewImage.style.display = "block";
        placeholderText.textContent = "Analyzing your photo...";

        try {
            await modelReady;

            if (thisUpload !== uploadVersion) {
                return;
            }

            detectFace();
        } catch (error) {
            if (thisUpload !== uploadVersion) {
                return;
            }

            console.error("Analysis failed:", error);

            measurementsSection.hidden = true;
            landmarkCanvas.style.display = "none";
            previewImage.style.display = "block";

            placeholderText.textContent =
                "Analysis could not finish. Refresh and try again.";

            placeholderText.style.display = "block";
        }
    };

    previewImage.onerror = function () {
        if (thisUpload !== uploadVersion) {
            return;
        }

        placeholderText.textContent =
            "This photo could not load. Try a JPG or PNG.";

        placeholderText.style.display = "block";
    };

    previewImage.src = currentImageURL;
});

// ========================================
// FACE DETECTION
// ========================================

function detectFace() {
    const results = faceLandmarker.detect(previewImage);

    if (!results.faceLandmarks.length) {
        measurementsSection.hidden = true;

        placeholderText.textContent =
            "No face detected. Try a clear, front-facing photo.";

        placeholderText.style.display = "block";
        return;
    }

    const landmarks = results.faceLandmarks[0];

    console.log("Number of landmarks:", landmarks.length);

    landmarkCanvas.width = previewImage.naturalWidth;
    landmarkCanvas.height = previewImage.naturalHeight;

    canvasContext.clearRect(
        0,
        0,
        landmarkCanvas.width,
        landmarkCanvas.height
    );

    canvasContext.drawImage(
        previewImage,
        0,
        0,
        landmarkCanvas.width,
        landmarkCanvas.height
    );

    drawLandmarks(landmarks);
    drawMeasurementLines(landmarks);

    const measurements = calculateFaceMeasurements(landmarks);

    calculateEyeTilt(landmarks);
    calculateEyeOpening(landmarks);

    if (measurements) {
        estimateFaceShape(measurements);
        measurementsSection.hidden = false;
    }

    previewImage.style.display = "none";
    placeholderText.style.display = "none";
    landmarkCanvas.style.display = "block";
}

// ========================================
// FACE AND EYE WIDTH MEASUREMENTS
// ========================================

function calculateFaceMeasurements(landmarks) {
    function distance(indexA, indexB) {
        return getLandmarkDistance(
            landmarks[indexA],
            landmarks[indexB]
        );
    }

    const faceWidth = distance(POINTS.cheekA, POINTS.cheekB);

    const faceHeight = distance(
        POINTS.foreheadTop,
        POINTS.chin
    );

    const foreheadWidth = distance(
        POINTS.foreheadA,
        POINTS.foreheadB
    );

    const jawWidth = distance(POINTS.jawA, POINTS.jawB);

    const rightEyeWidth = distance(
        POINTS.rightEyeOuter,
        POINTS.rightEyeInner
    );

    const leftEyeWidth = distance(
        POINTS.leftEyeOuter,
        POINTS.leftEyeInner
    );

    if (faceWidth <= 0) {
        return null;
    }

    const faceRatio = faceHeight / faceWidth;
    const foreheadRatio = foreheadWidth / faceWidth;
    const jawRatio = jawWidth / faceWidth;

    displayValue("face-width", `${faceWidth.toFixed(1)} px`);
    displayValue("face-height", `${faceHeight.toFixed(1)} px`);
    displayValue("face-ratio", faceRatio.toFixed(2));

    displayValue(
        "forehead-width",
        `${foreheadWidth.toFixed(1)} px`
    );

    displayValue("jaw-width", `${jawWidth.toFixed(1)} px`);

    displayValue(
        "forehead-ratio",
        `${(foreheadRatio * 100).toFixed(1)}%`
    );

    displayValue(
        "jaw-ratio",
        `${(jawRatio * 100).toFixed(1)}%`
    );

    displayValue(
        "right-eye-width",
        `${rightEyeWidth.toFixed(1)} px`
    );

    displayValue(
        "left-eye-width",
        `${leftEyeWidth.toFixed(1)} px`
    );

    displayValue(
        "right-eye-ratio",
        `${((rightEyeWidth / faceWidth) * 100).toFixed(1)}%`
    );

    displayValue(
        "left-eye-ratio",
        `${((leftEyeWidth / faceWidth) * 100).toFixed(1)}%`
    );

    return {
        faceRatio,
        foreheadRatio,
        jawRatio
    };
}

// ========================================
// EYE TILT
// ========================================

function calculateEyeTilt(landmarks) {
    const rightOuter = toPixels(landmarks[POINTS.rightEyeOuter]);
    const rightInner = toPixels(landmarks[POINTS.rightEyeInner]);

    const leftOuter = toPixels(landmarks[POINTS.leftEyeOuter]);
    const leftInner = toPixels(landmarks[POINTS.leftEyeInner]);

    const rightCenter = {
        x: (rightOuter.x + rightInner.x) / 2,
        y: (rightOuter.y + rightInner.y) / 2
    };

    const leftCenter = {
        x: (leftOuter.x + leftInner.x) / 2,
        y: (leftOuter.y + leftInner.y) / 2
    };

    const baselineX = leftCenter.x - rightCenter.x;
    const baselineY = leftCenter.y - rightCenter.y;
    const baselineLength = Math.hypot(baselineX, baselineY);

    if (baselineLength === 0) {
        displayValue("right-eye-tilt", "Unavailable");
        displayValue("left-eye-tilt", "Unavailable");
        return;
    }

    const horizontalX = baselineX / baselineLength;
    const horizontalY = baselineY / baselineLength;

    const upwardX = horizontalY;
    const upwardY = -horizontalX;

    function getTilt(inner, outer) {
        const eyeX = outer.x - inner.x;
        const eyeY = outer.y - inner.y;

        const rise = eyeX * upwardX + eyeY * upwardY;

        const run = Math.abs(
            eyeX * horizontalX + eyeY * horizontalY
        );

        return Math.atan2(rise, run) * 180 / Math.PI;
    }

    function formatTilt(angle) {
        const roundedMagnitude = Math.abs(angle).toFixed(1);

        if (Number(roundedMagnitude) === 0) {
            return "0.0° (level)";
        }

        const direction =
            angle > 0 ? "outer corner higher" : "outer corner lower";

        return `${roundedMagnitude}° (${direction})`;
    }

    displayValue(
        "right-eye-tilt",
        formatTilt(getTilt(rightInner, rightOuter))
    );

    displayValue(
        "left-eye-tilt",
        formatTilt(getTilt(leftInner, leftOuter))
    );
}

// ========================================
// EYE OPENING
// ========================================

function calculateEyeOpening(landmarks) {
    const rightWidth = getLandmarkDistance(
        landmarks[POINTS.rightEyeOuter],
        landmarks[POINTS.rightEyeInner]
    );

    const leftWidth = getLandmarkDistance(
        landmarks[POINTS.leftEyeOuter],
        landmarks[POINTS.leftEyeInner]
    );

    const rightHeight = getLandmarkDistance(
        landmarks[POINTS.rightEyeUpper],
        landmarks[POINTS.rightEyeLower]
    );

    const leftHeight = getLandmarkDistance(
        landmarks[POINTS.leftEyeUpper],
        landmarks[POINTS.leftEyeLower]
    );

    if (rightWidth <= 0 || leftWidth <= 0) {
        displayValue("right-eye-opening", "Unavailable");
        displayValue("left-eye-opening", "Unavailable");
        return;
    }

    displayValue(
        "right-eye-opening",
        `${((rightHeight / rightWidth) * 100).toFixed(1)}% of eye width`
    );

    displayValue(
        "left-eye-opening",
        `${((leftHeight / leftWidth) * 100).toFixed(1)}% of eye width`
    );
}

// ========================================
// PRELIMINARY FACE-SHAPE ESTIMATE
// ========================================

function estimateFaceShape(measurements) {
    const {
        faceRatio,
        foreheadRatio,
        jawRatio
    } = measurements;

    // Experimental starter rules, not validated shape cutoffs.
    let shape;

    if (foreheadRatio - jawRatio > 0.10) {
        shape = "Heart-like proportions";
    } else if (jawRatio - foreheadRatio > 0.10) {
        shape = "Triangle-like proportions";
    } else if (faceRatio >= 1.50) {
        shape = "Elongated proportions";
    } else if (faceRatio <= 1.30) {
        shape = "Round or soft oval proportions";
    } else {
        shape = "Oval-like proportions";
    }

    displayValue("face-shape", shape);
}

// ========================================
// DRAW LANDMARK DOTS
// ========================================

function drawLandmarks(landmarks) {
    canvasContext.save();

    canvasContext.fillStyle = "#c91f5d";

    const radius = Math.max(1, landmarkCanvas.width / 700);

    landmarks.forEach(function (landmark) {
        canvasContext.beginPath();

        canvasContext.arc(
            landmark.x * landmarkCanvas.width,
            landmark.y * landmarkCanvas.height,
            radius,
            0,
            Math.PI * 2
        );

        canvasContext.fill();
    });

    canvasContext.restore();
}

// ========================================
// DRAW MEASUREMENT LINES
// ========================================

function drawMeasurementLines(landmarks) {
    function drawLine(startIndex, endIndex, color) {
        const start = landmarks[startIndex];
        const end = landmarks[endIndex];

        canvasContext.beginPath();

        canvasContext.moveTo(
            start.x * landmarkCanvas.width,
            start.y * landmarkCanvas.height
        );

        canvasContext.lineTo(
            end.x * landmarkCanvas.width,
            end.y * landmarkCanvas.height
        );

        canvasContext.strokeStyle = color;
        canvasContext.stroke();
    }

    canvasContext.save();

    canvasContext.lineWidth = Math.max(
        2,
        landmarkCanvas.width / 400
    );

    drawLine(
        POINTS.foreheadA,
        POINTS.foreheadB,
        "#f59e0b"
    );

    drawLine(
        POINTS.cheekA,
        POINTS.cheekB,
        "#06b6d4"
    );

    drawLine(
        POINTS.jawA,
        POINTS.jawB,
        "#8b5cf6"
    );

    drawLine(
        POINTS.foreheadTop,
        POINTS.chin,
        "#22c55e"
    );

    drawLine(
        POINTS.rightEyeOuter,
        POINTS.rightEyeInner,
        "#ec4899"
    );

    drawLine(
        POINTS.leftEyeOuter,
        POINTS.leftEyeInner,
        "#ec4899"
    );

    canvasContext.restore();
}