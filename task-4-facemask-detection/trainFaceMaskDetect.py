from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
ap.add_argument("-p","--plot",default="plot.png")
ap.add_argument("-m","--model",type=str,default="mask_detector.model")
args = vars(ap.parse_args())

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

print("Loading Models")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data,dtype="float32")    
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size = 0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)



baseModel = MobileNetV2(weights="imagenet",include_top = False,input_tensor=Input(shape=(224,224,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS) 
model.compile(loss="binary_crossentropy", optimizer=opt,
metrics=["accuracy"])

print("training Head...")
H = model.fit(
    aug.flow(trainX,trainY, batch_size=BS),
    step_per_epoch=len(trainX) // BS,
    validation_data = (testX,testY),
    validation_steps = len(testX) // BS,
    epochs=EPOCHS
)

print("Evaluting Network....")
predIdxs = model.predict(testX, batch_size = BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_))

print("Saving MaskDetector model....")
model.save(args["model"], save_format="h5")