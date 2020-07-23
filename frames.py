import cv2



video_capture = cv2.VideoCapture("edit.mp4")
c = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    c += 1
    cv2.imshow('Video', frame)
    print(c)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()