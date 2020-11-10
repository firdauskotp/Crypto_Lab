import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.keras.backend import set_session
from flask import Flask, request
from flask_cors import CORS #pip install -U flask-cors
import cv2 #pip install opencv-python
import json
import numpy as np #pip install numpy
import base64
from datetime import datetime

graph = tf.compat.v1.get_default_graph()
app=Flask(__name__)
CORS(app)

s1=tf.compat.v1.Session()
set_session(s1)

mod=tf.keras.models.load_model('model/facenet_keras.h5', compile=False)
#Convert image to 128d
def img_to_encoding(path, model):
    img1 = cv2.imread(path, 1)
    img = img1[...,::-1]
    dim = (160, 160)
    # resize image
    if(img.shape != (160, 160, 3)):
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x_train = np.array([img])
    embedding = model.predict(x_train)
    return embedding


#sample database
database = {}
database["TestOne"] = img_to_encoding("images/ariftest.PNG",mod)

def verify(image_path, identity, database, model):
  
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding-database[identity])
    print(dist)
    if dist<5:
        print("It's " + str(identity) + ", welcome in!")
        match = True
    else:
        print("It's not " + str(identity) + ", please go away")
        match = False
    return dist, match


#function to return identity
def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 1000
    #Looping over the names and encodings in the database.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 5:
        print("Sorry, no access")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity

#route to verify user
app.route('/verify', methods=['GET','POST'])
def change():
    img_data = request.get_json()['image64']
    img_name = str(int(datetime.timestamp(datetime.now())))
    with open('images/'+img_name+'.jpg', "wb") as fh:
        fh.write(base64.b64decode(img_data[22:]))
    path = 'images/'+img_name+'.jpg'
    global s1
    global graph
    with graph.as_default():
        set_session(s1)
        min_dist, identity = who_is_it(path, database, mod)
    os.remove(path)
    if min_dist > 5:
        return json.dumps({"identity": 0})
    return json.dumps({"identity": str(identity)})

#route to register a user
@app.route('/register', methods=['GET','POST'])
def register():
    try:
        username = request.get_json()['username']
        img_data = request.get_json()['image64']
        with open('images/'+username+'.jpg', "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))
        path = 'images/'+username+'.jpg'

        global s1
        global graph
        with graph.as_default():
            set_session(s1)
            database[username] = img_to_encoding(path, mod)    
        return json.dumps({"status": 200})
    except:
        return json.dumps({"status": 500})

if __name__ == "__main__":
    app.run(debug=True)