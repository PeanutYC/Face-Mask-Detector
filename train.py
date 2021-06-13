# import necessary library
# for data augmentation, MobileNetV2 classifier, pre-processing, loading images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

# binarizing class labels, segmenting dataset, printing classification report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# to find and list images in dataset
from imutils import paths
# plot training curves
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# making it easier for CLI
ap = argparse.ArgumentParser()
# the path to input images with mask and w/out mask
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# path to the training history plot
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
# path to the resulting serialized face mask classification model
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to output face mask detector model")
args = vars(ap.parse_args())

# hyperparameters of the model
# initializing learning rate, no of epochs to train, batch size
LEARNING_RATE = 1e-4   # will have a learning rate decay schedule
EPOCHS = 10
BATCH_SIZE = 32

# loading and preprocessing images
# initialize the list of data and class images
print("Loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    # extract class label from the file name
    # so we can get "with_mask" or "without_mask" to get associated with the image
    label = imagePath.split(os.path.sep)[-2]
    # load the input image, resizing to (224,224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)  # scaling pixel intensities to range [-1, 1] for convenience
    # update data and label list
    data.append(image)
    labels.append(label)

# ensuring data and labels are in numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# (one-hot-encoding) on the labels
label_bin = LabelBinarizer()
labels = label_bin.fit_transform(labels)
labels = to_categorical(labels)

# splitting datasets into train and test sets
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
augmentation = ImageDataGenerator(rotation_range=20,
                                  zoom_range=0.15,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.15,
                                  horizontal_flip=True,
                                  fill_mode="nearest")

# loading pretrained MobileNetV2 model for face detection -> fine-tuning
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# defining the head of the model
# flatten and dropout layer to convert the data in 1D and prevent over-fitting
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(3, 3))(head_model)
head_model = Flatten(name='flatten')(head_model)   # transform 2D output of the base to 1D
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# CNN -> convolutional base (extract features) + dense head (determine class of the image)
# Our actual model
model = Model(inputs=base_model.input, outputs=head_model)


# we are only using the basic model, so we will keep the layers frozen and only modify the last layer
# weights of the base layers will not be updated during backpropagation
# only head layers weight will be tuned
for layer in base_model.layers:
    layer.trainable = False

# compiling model
print("Compiling model...")
# learning rate decay schedule
optimizer = Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

# train model
print("Training head...")

head = model.fit(augmentation.flow(X_train, y_train, batch_size=BATCH_SIZE),
                 steps_per_epoch=len(X_train) // BATCH_SIZE,
                 validation_data=(X_test, y_test),
                 validation_steps=len(X_test) // BATCH_SIZE,
                 epochs=EPOCHS)

# evaluate model performance on test set
print("Evaluating network...")
# make predictions on test set
predicted_labels = model.predict(X_test, batch_size=BATCH_SIZE)
# For each image in the test set, we need to find the
# index of the label with corresponding largest predicted probability
# By adding the axis argument, numpy looks at the rows and columns individually
# axis=1 -> operation is performed across the row of result
predicted_labels = np.argmax(predicted_labels, axis=1)
# print classification report
print(classification_report(y_test.argmax(axis=1), predicted_labels, target_names=label_bin.classes_))

# Save model (serialize model to disk)
print("Saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot training report for accuracy and loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), head.history["loss"], label="Training loss")
plt.plot(np.arange(0, EPOCHS), head.history["val_loss"], label="Validation loss")
plt.plot(np.arange(0, EPOCHS), head.history["binary_accuracy"], label="Training accuracy")
plt.plot(np.arange(0, EPOCHS), head.history["val_binary_accuracy"], label="Validation accuracy")
plt.title("TRAINING REPORT: LOSS & ACCURACY")
plt.ylabel("Loss/Accuracy")
plt.xlabel("Epoch number")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


