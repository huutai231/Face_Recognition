import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-da1-default-rtdb.firebaseio.com/"
})

ref = db.reference('Person')

data = {
    "21119": {
        "name": "Nguyen Huu Tai",
        "position": "Student",
        "last_attendance_time": "2024-3-26 13:40:00"
    },

    "12345": {
            "name": "Mark Zuckerberg",
            "position": "Engineer",
            "last_attendance_time": "2024-3-26 13:47:20"
        },

    "32412": {
            "name": "Lionel Messi",
            "position": "Soccer",
            "last_attendance_time": "2024-3-26 14:23:11"
        }
}
for key, value in data.items():
    ref.child(key).set(value)

