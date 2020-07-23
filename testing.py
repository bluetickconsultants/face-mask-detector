from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

model=load_model('mnm_2.h5')

def testing_image():

  # load the image
  image = cv2.imread("1.jpg",0)
  # orig = image.copy()
  orig = cv2.imread("1.jpg",1)
  print(image.shape)
  cascPath = "haarcascade_frontalface_default.xml"
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
  
  # pre-process the image for classification
      image = cv2.resize(image, (222, 222))
      print("resize", image.shape)
      image = image.astype("float32") / 255.0
      print("astype", image.shape)
      image = img_to_array(image)
      print("imgtoarray", image.shape)
      image = np.expand_dims(image, axis=0)
      print("expanddims", image.shape)



      predictions=model.predict(image)
      print(predictions[0])
  label = ['Mask','Nomask',]

  label_nomask = "{}: {:.2f}%".format(label[1], predictions[0][0] * 100)
  label_mask = "{}: {:.2f}%".format(label[0], predictions[0][1] * 100)
    
  # # draw the label on the image
  output = imutils.resize(orig, width=400)
  if predictions[0][0] > 0.6 or predictions[0][1] < 0.65:
    cv2.putText(output, label_nomask, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
      0.6, (0, 0, 255), 1)
    cv2.putText(output, "ALERT!!!", (200, 200),  cv2.FONT_HERSHEY_SIMPLEX,
      1, (0, 0, 255), 2)
  else:

    cv2.putText(output, label_mask, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
      0.6, (0, 255, 0), 1)
    cv2.putText(output, "SAFE", (200, 200),  cv2.FONT_HERSHEY_SIMPLEX,
      1, (0, 255, 0), 4)
    
  # show the output image
  cv2.imshow("Output", output)
  cv2.waitKey(0)

  cv2.destroyAllWindows()
  cv2.waitKey(1)


def testing_video():
  cascPath = "haarcascade_frontalface_default.xml"
  faceCascade = cv2.CascadeClassifier(cascPath)

  video_capture = cv2.VideoCapture('vid1.mov')
  c = 0

  while True:
      # Capture frame-by-frame
      ret, frame = video_capture.read()

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      faces = faceCascade.detectMultiScale(
          gray,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30, 30),
          flags=cv2.CASCADE_SCALE_IMAGE
      )

      # Draw a rectangle around the faces
      for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
          image = frame[y:y + h, x:x + w]
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          image = cv2.resize(image, (222, 222))
          image = image.astype("float32") / 255.0
          image = img_to_array(image)
          image = np.expand_dims(image, axis=0)



          predictions=model.predict(image)
          print(np.squeeze(predictions))
      try:
          label = ['Mask','Nomask']
          label_nomask = "{}: {:.2f}%".format(label[1], predictions[0][0] * 100)
          label_mask = "{}: {:.2f}%".format(label[0], predictions[0][1] * 100)
            
          # # draw the label on the image
          if predictions[0][0] > 0.6 or predictions[0][1] < 0.65:
            cv2.putText(frame, label_nomask, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
              0.6, (0, 0, 255), 1)
            cv2.putText(frame, "ALERT!!!", (200, 200),  cv2.FONT_HERSHEY_SIMPLEX,
              1, (0, 0, 255), 2)
          else:

            cv2.putText(frame, label_mask, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
              0.6, (0, 255, 0), 1)
            cv2.putText(frame, "SAFE", (200, 200),  cv2.FONT_HERSHEY_SIMPLEX,
              1, (0, 255, 0), 4)

              # Display the resulting frame
          cv2.imshow('Video', frame)
          if (cv2.waitKey(1) & 0xFF == ord('q')):
              break
      except:
          pass

  # When everything is done, release the capture
  video_capture.release()
  cv2.destroyAllWindows()




# testing_image()
testing_video()

#222.86319218241042