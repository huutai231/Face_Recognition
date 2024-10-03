import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import matplotlib.pyplot as plt

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-da1-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-da1.appspot.com"
})
bucket = storage.bucket()


# Read background and Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 580)
cap.set(4, 440)
imgBackground = cv2.imread("../Resource/Background.png", 1)


# import modes image into imgModelist
folderModePath = "../Resource/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))


# import the coding file
print("Loading encode file")
file = open("EncodeFile.p", "rb")
encodeListWithIds = pickle.load(file)
print("Loading succesfully")
file.close()
encodeListKnow, IdsList = encodeListWithIds

counter = 0
id = -1
imgPerson = []
# -------------------------------------
while True:
    success, img = cap.read()
    img = cv2.resize(img, (580, 440))


    imgSmall = cv2.resize(img, (0, 0), None, 0.2, 0.2)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    faceCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurrFrame = face_recognition.face_encodings(imgSmall, faceCurrFrame)
    # encodeCurrFrame = []
    imgBackground[70:70+440, 30:30+580] = img
    modeType = 3
    imgBackground[0:540, 640:960] = imgModeList[modeType]

    for encodeFace, faceLocation in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print("matches: ", matches)
        print("Face distance: ", faceDis)

        matchIndex = np.argmin(faceDis)
        y0, x1, y1, x0 = faceLocation
        y0, x1, y1, x0 = y0 * 5, x1 * 5, y1 * 5, x0 * 5
        bbox = 30 + x0, 70 + y0, x1 - x0, y1 - y0
        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

        if (matches[matchIndex] and faceDis[matchIndex] <= 0.4):
            #print("Known face detected: ", IdsList[matchIndex])
            id = IdsList[matchIndex]
            if counter == 0:
                counter = 1
                modeType = 1
        else:
            modeType = 0
            imgBackground[0:540, 640:960] = imgModeList[modeType]

    if counter != 0:
        if counter == 1:
            # Get the data
            personInfo = db.reference(f'Person/{id}').get()
            print(personInfo)

            # Get the images from storage
            blob = bucket.get_blob(f'../Images/{id}.png')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgPerson = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            imgPerson = cv2.resize(imgPerson, (260, 350))

            # Update data
            ref = db.reference(f'Person/{id}')
            datetimeObject = datetime.strptime(personInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
            seconds = (datetime.now()-datetimeObject).total_seconds()
            ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if 10 < counter < 20:
            modeType = 2
        imgBackground[0:540, 640:960] = imgModeList[modeType]
        if counter <= 10:
            (w, h), _ = cv2.getTextSize(personInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            offset = (320 - w) // 2
            cv2.putText(imgBackground, str(personInfo['name']), (640 + offset, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            imgBackground[40:40+350, 670:670+260] = imgPerson

        counter += 1
        if counter >= 20:
            counter = 0
            modeType = 3
            personInfo = []
            imgPerson = []
            imgBackground[0:540, 640:960] = imgModeList[modeType]
    cv2.imshow("Face Recognition", imgBackground)
    cv2.waitKey(1)