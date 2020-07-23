import cv2
import os

cascPath = "haarcascade_frontalface_default.xml"

def testing_image(filepath):

  # load the image
  image = cv2.imread(filepath)
  # orig = image.copy()
  
  faceCascade = cv2.CascadeClassifier(cascPath)
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(
      image,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags=cv2.CASCADE_SCALE_IMAGE
  )

  # Draw a rectangle around the faces
  for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
      roi_color = image[y:y + h, x:x + w]
      return roi_color
  

    
  # show the output image
  # cv2.imshow("Output", image)
  # cv2.waitKey(0)

  # cv2.destroyAllWindows()
  # cv2.waitKey(1)


path = "facemasks"

files = os.listdir(path)
c = 0
for file in files:
  filepath = os.path.join(path, file)
  face = testing_image(filepath)
  if face is not None:

    cv2.imwrite("temp_data/" + str(c) + 'face.jpg', face)
    c += 1

  