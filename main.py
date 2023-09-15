import cv2 as cv


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)  # Corrected dimensions tuple
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    frame_resized = rescaleFrame(frame)
    img = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('resize video', frame_resized)

    # Close the video when 'd' is pressed
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
