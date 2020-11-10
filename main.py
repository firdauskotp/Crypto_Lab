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

mod=tf.keras.models.load_model('model/facenet_keras.h5')
