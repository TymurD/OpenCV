import os
import numpy as np
import cv2

face_net = cv2.dnn.readNetFromCaffe(
    "data/DNN/deploy.prototxt",
    "data/DNN/res10_300x300_ssd_iter_140000.caffemodel"
)
eye_cascade = cv2.CascadeClassifier(
    "data/haarcascade/haarcascade_eye.xml"
)

INPUT_DIRECTORY = "bulk-image"
OUTPUT_DIRECTORY = "bulk-result"
FORMATS = ('.jpg', '.jpeg', '.png', ".webp")

os.makedirs(INPUT_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
files = sorted(os.listdir(INPUT_DIRECTORY))

for file in files:
    if not file.lower().endswith(FORMATS):
        continue

    path = os.path.join(INPUT_DIRECTORY, file)
    img = cv2.imread(path)
    if img is None:
        continue

    original_image = cv2.imread(path)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    (h, w) = original_image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        original_image, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x2, y2) = box.astype("int")
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(original_image, (x, y), (x2, y2), (0, 255, 0), 2)

        roi_gray = gray[y:y2, x:x2]
        roi_color = original_image[y:y2, x:x2]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                            minNeighbors=10,
                                            minSize=(10, 10))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2
            )

    output_path = os.path.join(OUTPUT_DIRECTORY, file)
    cv2.imwrite(output_path, original_image)
