import cv2 as cv
import os
import time

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(base_dir, "models")
logs_dir = os.path.join(base_dir, "logs")

os.makedirs(logs_dir, exist_ok=True)

recognizer_path = os.path.join(model_dir, "recognizer.yml")
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(recognizer_path)

# Load the label-to-name mapping
label_map = {}
labels_path = os.path.join(model_dir, "labels.txt")
with open(labels_path, "r") as f:
    for line in f:
        label, name = line.strip().split(":")
        label_map[int(label)] = name

target_people = ["antoine"]

# Load Haar Cascade face detector
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv.VideoCapture(0)
alert_triggered = False
last_alert_time = 0

print("🔍 Face detection started. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv.resize(face_roi, (200, 200))

        label, confidence = recognizer.predict(face_roi)
        name = label_map.get(label, "Unknown")

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Alert if target person is recognized with good confidence
        if name in target_people and confidence < 80:
            now = time.time()
            if now - last_alert_time > 5:  # prevent repeated alerts
                print(f"[ALERT] {name} detected!")

                alert_log_path = os.path.join(logs_dir, "alert_log.txt")
                with open(alert_log_path, "a") as log:
                    log.write(f"{time.ctime()} - {name} detected.\n")

                last_alert_time = now
                alert_triggered = True

    cv.putText(frame, f"{len(faces)} face(s) detected", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.imshow("Face Detection + Alert", frame)
    if cv.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv.destroyAllWindows()
