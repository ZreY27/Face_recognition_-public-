import cv2 as cv
import time

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv.VideoCapture(0) #if the webcam is plug to the PC, use 1 instead

if not cap.isOpened():
    print(" /!\ Webcam not found /!\ ")
    exit()

print("Press ESCAPE to quit.") #ESCAPE = ÉCHAP (fr)

prev_time = time.time()

while True:
   
    ret, frame = cap.read()
    if not ret:
        print(" /!\ Webcam can't be used /!\ ")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        cv.rectangle(
            frame, (x, y), (x + w, y + h), (0, 255, 0), 2
        )

    text = f"{len(faces)} face(s) detected."
    cv.putText(
        frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_text = f"FPS : {int(fps)}"
    cv.putText(
        frame, fps_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    cv.imshow("Live detection (Press ESCAPE to quit).", frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27: # ESCAPE key / Touche ECHAP
        break

cap.release()
cv.destroyAllWindows()
