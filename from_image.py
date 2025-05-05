import cv2
import os

# Define paths
face_cascade_path = r"C:\Users\kdine\Downloads\haarcascade_frontalface_alt.xml"
image_path = r"C:\Users\kdine\Downloads\WhatsApp Image 2025-02-09 at 9.14.30 PM.jpeg"

# Verify that the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}. Check the path.")
    exit()

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

# Load the image (keep original size)
img = cv2.imread(image_path)

# Check if image is loaded properly
if img is None:
    print("Error: Image could not be read. Check file format and integrity.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Improve contrast using histogram equalization
gray = cv2.equalizeHist(gray)

# Detect faces with optimized settings
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

# Print number of faces detected
print(f"Number of faces detected: {len(faces)}")

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Ensure the window can be resized to fit the full image
cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)

# Display the full image
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
