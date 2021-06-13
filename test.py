# Import necessary packages
# load our mask net model and preprocessing the input image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import argparse
# display and manipulate image
import cv2

# making it easier for CLI
ap = argparse.ArgumentParser()
# path to input image for testing
ap.add_argument("-i", "--image", required=True, help="path to input image")
# localize faces prior to classifying them
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
# path to our trained model
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model")
# optional probability threshold can be set to override 50% to filter weak face detections
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Loading face detector model(from disk) for ROI(Region of Interest) identification
print("Loading face detector model...")

# load serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loading our trained model for classification
print("Loading face mask detector model...")
model = load_model(args["model"])

# preprocessing the image
# read the image
image = cv2.imread(args["image"])
# make a copy of the original image
orig = image.copy()
# frame dimensions -> grab image spatial dimensions
(h, w) = image.shape[:2]
# construct a blob that can be passed through the pre-trained image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 213.0))  # resizing to 300x300 pixels

print("Computing face detections...")
# set the input to the pre-trained deep learning network and obtain the output predicted probabilities
net.setInput(blob)
# perform face detection to localize where are the faces in the image
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence associated with the detections
    confidence = detections[0, 0, i, 2]
    # carry forward only if the confidence in face detection is greater than our threshold value
    if confidence > args["confidence"]:
        # computing x,y coordinate for bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # ensure bounding box lies within dimensions of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w-1, endX), min(h-1, endY))
        # extracting face ROI via slicing
        face = image[startY:endY, startX:endX]
        # convert it from BGR to RGB channel ordering
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        # passing the face through our model to predict whether a mask is present or not
        (mask, withoutMask) = model.predict(face)[0]

        # determine the label and the colour of the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # green for with_mask, red for without_mask
        # include probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)

        # 1st parameter is img, 2nd is text, 3rd is bottom left (where the text starts), 4th is font type,
        # 5th is font size, 6th is colour, 7th is thickness of text
        cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        # 1st parameter is img, 2nd is top-left corner, 3rd is bottom right, 4th is colour, 5th is thickness
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
