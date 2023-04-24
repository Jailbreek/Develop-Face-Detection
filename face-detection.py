import cv2

cam = cv2.VideoCapture(1)# create a VideoCapture object that reads video the camera
cam.set(3, 660)# set the width
cam.set(4, 500)# set the height 
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')# create Cascade Classifier model that used to detect face

while True:
    retV, frame = cam.read()# read the Frame from camera
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# convert the frame to GrayScale
    faces = faceDetector.detectMultiScale(color, 1.2, 5)# coordinates of the bounding box around the face
    
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 63), 2)# draw a rectangle around the face in the face 
        rec_face = color[y : y + h, x : x + w]# extract the ROI that corresponds to the face

        # calculate coordinates of text
        text_x = x + int(w / 2.3)
        text_y = y + h + 20  # add 20 pixels to the y-coordinate to place text below the bounding box

        # draw text on frame
        cv2.putText(frame, 'face', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 225, 63), 2)
    
    cv2.imshow('WEBCAM', frame)# calling Camera frame
    close = cv2.waitKey(1) & 0xFF# declare exit variable
    if close == 27 or close == ord('n'):# conditional check for exit
        break
    
cam.release()
cv2.destroyAllWindows()
