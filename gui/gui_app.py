import os
import cv2 as cv
import numpy as np
from tkinter import Tk, Button, filedialog, Canvas
from PIL import Image, ImageTk

# --- Chemins vers le modèle et les labels ---
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RECOGNIZER_PATH = os.path.join(MODEL_DIR, "recognizer.yml")
LABELS_PATH     = os.path.join(MODEL_DIR, "labels.txt")

# --- Chargement du recognizer LBPH et du mapping labels → noms ---
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(RECOGNIZER_PATH)

label_map = {}
with open(LABELS_PATH, "r") as f:
    for line in f:
        idx, name = line.strip().split(":")
        label_map[int(idx)] = name

# --- Classificateur Haar pour la détection ---
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class App:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition GUI")

        # Canvas pour afficher l'image
        self.canvas = Canvas(root, width=600, height=450)
        self.canvas.pack()

        # Boutons
        self.btn_load = Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack(side="left", padx=10, pady=10)

        self.btn_detect = Button(root, text="Detect & Recognize", command=self.detect_faces)
        self.btn_detect.pack(side="right", padx=10, pady=10)

        self.image_path = None
        self.photo = None  # pour garder la référence PhotoImage

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
        if not path:
            return
        self.image_path = path
        # Charger et redimensionner pour l’affichage
        img = Image.open(path)
        img.thumbnail((600, 450), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def detect_faces(self):
        if not self.image_path:
            return
        # Lire avec OpenCV
        frame = cv.imread(self.image_path)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Traiter chaque visage
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv.resize(roi, (200, 200))
            label, conf = recognizer.predict(roi)
            name = label_map.get(label, "Unknown")

            # Dessiner rectangle + texte
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame, f"{name}", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Convertir BGR→RGB puis pour Tkinter
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil.thumbnail((600, 450), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)


if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
