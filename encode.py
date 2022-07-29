# This file encodes faces in the known faces directory
# importing usefull moduels
import pickle
import cv2
import os
import face_recognition

# creating empty list for images and names
images = []
classNames = []
# specifing image path
path = "KNOWN_FACES"
myList = os.listdir(path)
Identified = False

# creating a function to finding faces encoding


def findEncodings(images):
    # reading every images in out specified path
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeList = []

    # Changing images to RGB and encoding the faces
    # encode = face_recognition.face_encodings(img)
    # if len(encode) > 0:
    #     encodelist.append(encode[0])
    for img, filename in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
        # saving the encodings
        with open("data.pickle", "wb") as f:
            pickle.dump(encodeList, f)


def encodeListKnown():
    findEncodings(images)
