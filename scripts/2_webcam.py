import cv2 as cv
import os

save_dir = "dataset/antoine"
os.makedirs(save_dir, exist_ok=True)

def get_next_filename(base_dir, prefix="photo_", extension=".jpg"):
    i = 0
    while True:
        filename = f"{prefix}{i}{extension}"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1

cap = cv.VideoCapture(0) #if the webcam is plug to the PC, use 1 instead

if not cap.isOpened():
    print(" /!\ Webcam not found /!\ ")
    exit()

print("Press SPACE to take a screenshot.")
print("Press ESCAPE to quit.") #ESCAPE = ÉCHAP (fr)

while True:

    ret, frame = cap.read()
    if not ret:
        print(" /!\ Webcam can't be used /!\ ")
        break

    cv.imshow("Webcam - Press SPACE to take a screenshot.", frame)
   
    key = cv.waitKey(1) & 0xFF

    if key == 27:  # ESCAPE key / Touche ECHAP
        break
    elif key == 32:  # SPACE key / Touche ESPACE
        filepath = get_next_filename(save_dir)
        cv.imwrite(filepath, frame)
        print(f"📷 saved in {filepath}")
        
cap.release()
cv.destroyAllWindows()
