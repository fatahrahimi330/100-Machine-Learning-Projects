# Import required libraries
import cv2
import numpy as np


# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not access camera. Please grant camera permissions.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Capture frames continuously
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame was captured successfully
    if not ret or frame is None:
        print("Error: Could not read frame from camera.")
        break
    
    frame = cv2.flip(frame, 1)

    # RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterator to count faces
    i = 0
    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Increment iterator for each face in faces
        i = i+1

        # Display the box and faces
        cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print('Face', i)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # This command let's us quit with the "q" button on a keyboard.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()