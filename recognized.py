# importing usefull moduels
import pickle
import numpy as np
import cv2
import os
import face_recognition


# creating a function to recognize faces

def recognize(filename):
    model_results = "NOT IDENTIFIED"
    # creating empty list for images and names
    images = []
    classNames = []

    # specifing image path
    path = "known_faces"
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    # loading the saved encodings
    with open("data.pickle", "rb") as f:
        encodeList = pickle.load(f)
    # Reading image file
    img = cv2.imread(filename)

    # Converting image or RGB and finding faces encoding and locations
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)

    # Compering encodings
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        match = face_recognition.compare_faces(encodeList, encodeFace)
        faceDis = face_recognition.face_distance(encodeList, encodeFace)
        # Finding highest match
        matchIndex = np.argmin(faceDis)

        # Matching face and timestamps
        if match[matchIndex]:
            print(match)
            if faceDis[matchIndex] < 0.45:
                name = classNames[matchIndex].upper()
                model_results = name
                # Drawing rectangles around faces and writing timestamps
                top_left = (faceLoc[3], faceLoc[0])
                bottom_right = (faceLoc[1], faceLoc[2])

                # Get color by name using our fancy function
                color = (0, 255, 0)

                # Paint frame
                cv2.rectangle(img, top_left, bottom_right, color, 3)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (faceLoc[3], faceLoc[2])
                bottom_right = (faceLoc[1], faceLoc[2] + 22)

                # Paint frame
                cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(img, name + ' seconds', (faceLoc[3] + 10, faceLoc[2] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 250), 2)
            else:
                model_results = "NOT IDENTIFIED"

    cv2.imshow('Live feed', img)
    cv2.waitKey(0)

    return model_results


