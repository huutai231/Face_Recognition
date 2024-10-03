import os
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-da1-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-da1.appspot.com"
})


# import face images and Id
folderImagesPath = "../Images"
imagesPathList = os.listdir(folderImagesPath)
imgList = []
IdsList = []
for path in imagesPathList:
    imgList.append(cv2.imread(os.path.join(folderImagesPath, path)))
    IdsList.append(os.path.splitext(path)[0])

    # Give images to Database

    fileName = f'{folderImagesPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


# find encoding
def findEncoding(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


print("ENCODING BEGIN")
encodeListknown = findEncoding(imgList)
encodeListKnowWithId = [encodeListknown, IdsList]
print("ENCODING COMPLETE")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnowWithId, file)
file.close()
print("File saved")