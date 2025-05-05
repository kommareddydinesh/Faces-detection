# Face Detection with OpenCV using Haar Cascades
This project demonstrates face detection using OpenCV and a pre-trained Haar Cascade classifier on static images and webcam input.

## 📂 Project Structure
**Make sure**
📁 DATA SCIENCE (project root) 
├── from_image.py # Detects faces from an image
├── live.py # Detects faces from live webcam stream
├── one_time.py # Detects faces from a single webcam frame
├── haarcascade_frontalface_alt.xml # Haar cascade XML classifier

## 🧠 About the Classifier
The `haarcascade_frontalface_alt.xml` file is a pre-trained Haar Cascade classifier for detecting frontal human faces. It is widely used with OpenCV for basic face detection tasks.

## 🚀 How It Works
- Load the Haar cascade model.
- Read the input (either image or webcam frame).
- Convert to grayscale and apply histogram equalization.
- Detect faces using `detectMultiScale`.
- Draw rectangles around detected faces.
- Display the result.


## 🖼️ Example (Static Image)
```python
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))```


****🖥️ Requirements****
Python 3.x
OpenCV (cv2)
Webcam (for live/demo usage)

Install OpenCV:
---In bash/Terminal---
pip install opencv-python

****▶️ Usage****
Detect from image:
---In bash/Terminal---
python from_image.py
Detect from webcam (live stream):
---In bash/Terminal---
python live.py
Detect from a single webcam frame:
---In bash/Terminal---
python one_time.py

****📌 Notes****
Make sure haarcascade_frontalface_alt.xml is present in your project directory.
Adjust parameters like scaleFactor and minNeighbors for improved detection accuracy.
