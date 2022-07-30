# This file analize the video, detects faces and saves them
# importing usefull moduels
import cv2
import mediapipe as mp
import os

# Creating folders
directory = "known_faces"

# try:
#     # making directory
#     path = os.path.join(os.getcwd(), directory)
#     os.mkdir(path)
# except EOFError:
#     pass


# creating a function to read video


def run(filename):
    # reading video
    cap = cv2.VideoCapture(filename)
    frame = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # detecting faces
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    while True:
        # counting framerate
        frame_count += 1
        timeInS = float(frame_count) / frame
        # loading faces
        success, img = cap.read()
        if success:
            #  Converting and resizing video
            imgResize = cv2.resize(img, (1000, 500))
            imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(imgRGB)
            # Crop out every detected faces and save every faces frame
            if results.detections:
                for id, detection in enumerate(results.detections):
                    bboxClass = detection.location_data.relative_bounding_box
                    iH, iW, iC = imgResize.shape
                    bbox = int(bboxClass.xmin * iW), int(bboxClass.ymin * iH), \
                           int(bboxClass.width * iW), int(bboxClass.height * iH)
                    cv2.rectangle(imgResize, bbox, (225, 0, 255), 2)
                    cv2.putText(imgResize, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    # Save frames
                    cv2.imwrite('known_faces/' +
                                str(round(timeInS, 2)) + '.png', img)
                    

            if cv2.waitKey(0) & 0xFF == 1:
                break
        else:
            break
