# Face Recognition Project / Projet de Reconnaissance Faciale

This project demonstrates a full face recognition pipeline in Python.\
Ce projet présente une chaîne complète de reconnaissance faciale en Python.

---

## 📦 Project Structure / Structure du projet

```
face_recognition_project/
├── dataset/               # Raw images per person / Images brutes par personne
├── models/                # Trained LBPH model and labels / Modèle LBPH entraîné et labels
├── logs/                  # Alert logs / Journaux d'alertes
├── scripts/               # Python scripts by step / Scripts Python par étape
│   ├── 1_face_detection.py        # Detect faces on an image / détecte les visages sur une image
│   ├── 2_webcam.py                # Capture webcam images / capture images webcam
│   ├── 3_face_detection_on_webcam.py  # Live detection / détection en direct
│   ├── 4_alert_system.py          # Alert on recognition / système d'alerte
│   └── 4_train_model.py           # Train model for alert system / Entraîner le modèle pour le système d'alerte
├── gui/                   # GUI application (Tkinter) / application GUI
│   └── gui_app.py
├── docs/
│   └── screenshots/       # Demo screenshots / Captures d'écran démo
├── README.md              # Bilingual project README / README bilingue
└── requirements.txt       # Python dependencies / Dépendances Python
```

## ⚙️ Installation / Installation

1. **Clone** the repository:\
   **Cloner** le dépôt :
   ```bash
   git clone https://github.com/yourusername/face_recognition_project.git
   cd face_recognition_project
   ```
2. **Create and activate** a Python virtual environment:\
   **Créer et activer** un environnement virtuel Python :
   ```bash
   python -m venv env
   source env/bin/activate    # macOS/Linux
   env\Scripts\activate     # Windows
   ```
3. **Install** dependencies:\
   **Installer** les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage / Utilisation

### 1. Capture images / Capturer des images

```bash
python scripts/1_capture_images.py
```

### 2. Train the model / Entraîner le modèle

```bash
python scripts/2_train_model.py
```

### 3. Live recognition / Reconnaissance en direct

```bash
python scripts/3_realtime_recognition.py
```

### 4. Alert system / Système d'alerte

```bash
python scripts/4_alert_system.py
```

### 5. GUI application / Application GUI

```bash
python gui/gui_app.py
```

## 📝 License / Licence

**All rights reserved**. This project is proprietary and **cannot be shared or modified** without explicit permission from the author.

Tout droits réservés. Ce projet est propriétaire et **ne peut ni être partagé ni modifié** sans autorisation expresse de l'auteur.

