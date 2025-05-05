import cv2
import os

# Define path for Haar cascade
face_cascade_path = r"C:\Users\kdine\Downloads\haarcascade_frontalface_alt.xml"

# Verify that the Haar cascade file exists
if not os.path.exists(face_cascade_path):
    print(f"Error: Haar cascade file not found at {face_cascade_path}. Check the path.")
    exit()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if Haar cascade is loaded correctly
if face_cascade.empty():
    print("Error: Could not load Haar cascade. Check the file integrity.")
    exit()

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Allow camera to warm up
cv2.waitKey(2000)  # Wait for 2 seconds to adjust lighting

# Capture a single frame
ret, frame = cap.read()

# Release the webcam after capturing the frame
cap.release()

if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

# Convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Improve contrast using histogram equalization
gray = cv2.equalizeHist(gray)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

# Print the number of faces detected
face_count = len(faces)
print(f"Number of faces detected: {face_count}")

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the captured frame with detected faces
cv2.imshow("Detected Faces", frame)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
