import cv2 as cv

def face_detection(image_path):
    image = cv.imread(image_path)
    if image is None:
        print(" /!\ Image not found /!\ ")
        return
   
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8, 
        minSize=(100, 100) 
    )

    print(f"face count : {len(faces)}")

    str_detected = "face detected"
    if len(faces) != 0:

        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    else:
        str_detected = "face not detected"

    display_image = cv.resize(image, (800, 800))
    cv.namedWindow(str_detected, cv.WINDOW_NORMAL)
    cv.imshow(str_detected, display_image)   
    cv.waitKey(0)
    cv.destroyAllWindows()

face_detection("dataset/notface1.jpg")
face_detection("dataset/antoine/face1.jpg")
