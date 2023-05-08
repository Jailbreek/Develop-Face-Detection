import cv2

# Create a VideoCapture object that reads video from the camera
camera = cv2.VideoCapture(0)

# Set the width and height of the camera frames
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 660)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# Load the face detection cascade classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces():
    """
    Continuously read frames from the camera, detect faces in each frame,
    and draw a rectangle around each face.
    """
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()

        # Checking if the frame was successfully read from the camera
        if not ret:
            print("Error: Cannot read from camera")
            break

        try:
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # If the frame cannot be converted to grayscale, this will skip
            print("Error: Cannot convert frame to grayscale")
            continue

        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(
            gray_frame, scaleFactor=1.2, minNeighbors=5)

        # Draw a rectangle around each detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 63), 2)
            text_x = x + int(w / 2.3)
            text_y = y + h + 20
            cv2.putText(frame, 'face', (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 225, 63), 1)

        # Display the frame with the detected faces
        cv2.imshow('Face-Detect', frame)

        # Check if the user wants to exit the program
        close = cv2.waitKey(1) & 0xFF
        if close == 27 or close == ord('n'):
            break

    # Release the camera and destroy all windows
    camera.release()
    cv2.destroyAllWindows()


# Call the detect_faces function to start the face detection program
detect_faces()
