import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

import cv2
import pandas as pd
import numpy  as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from extract_bottleneck_features import *
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
graph = tf.get_default_graph()

from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob

import csv
from itertools import chain

with open('data/dog_names.csv', 'r') as f:
  reader = csv.reader(f)
  dog_names = list(reader)

dog_names=list(chain.from_iterable(dog_names))

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50  = bottleneck_features['test']

### TODO: Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.summary()

### TODO: Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('models/weights.best.Resnet50.hdf5')

def path_to_tensor(img_path):
    """
    This function takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.

    Parameter:
    img_path: the path of the dog breeds project dataset
    
    Returns:
    a 4D tensor with shape (1,224,224,3) suitable for supplying to a Keras CNN
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    """
    This function takes a string-valued file path to a color image as input and returns prediction vector for image located at img_path

    Parameter:
    img_path: the path of the dog breeds project dataset
    
    Returns:
    prediction vector for image located at img_path
    """
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    """
    This function takes a path to an image as input and returns True or False representing whether a face is detected in the image or not

    Parameter:
    img_path: the path of the image user wants to identificate the possible dog breed
    
    Returns:
    True or False representing whether a face is detected in the image or not
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    """
    This function takes a path to an image as input and returns True or False representing whether a dog is detected in the image or not

    Parameter:
    img_path: the path of the image user wants to identificate
    
    Returns:
    True or False representing whether a dog is detected in the image or not
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def predict_breed(img_path):
    """
    This function takes a path to an image as input and returns the dog breed that is predicted by the model.

    Parameter:
    img_path: the path of the image user wants to identificate the possible dog breed
    
    Returns:
    The possible dog breed (name from the dog_names list) predicted
    """
    return dog_names[np.argmax(Resnet50_model.predict(extract_Resnet50(path_to_tensor(img_path))))]

def fun_app(img_path):
    """
    This function return the possible dog breed of the dog/human in the input image

    Parameter:
    img_path: the path of the image user wants to identificate the possible dog breed
    
    Returns:
    A message indicates the possible dog breed of the dog/human in the input image
    """
    if dog_detector(img_path) == True:
        Response = "Hello, lovely dog, you belong to ...%s" %predict_breed(img_path)
    elif face_detector(img_path) == True:
        Response = "Hello, human, you look like a ...%s" %predict_breed(img_path)
    else:
        Response = "Error, not human or dog detected!"
    return Response

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
ALLOWED_EXTENSIONS = set(['JPG', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit upload size upto 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('send_file', filename=filename)
            file_for_app = '/home/workspace/CapstoneProject/uploads/'+filename
            global graph
            with graph.as_default():
                Result_Message = fun_app(file_for_app)
            flash(Result_Message)
            return redirect(url_for('uploaded_file', filename=filename, Result_Message=Result_Message))
    return render_template('go.html')

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # use model to find the dog breed
    query = 'test'
    #query  = url_for('send_file', filename=filename)
    Result_Message=filename
    print("AAAA")
    print(Result_Message)
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        filename=filename,
        Result_Messsage=filename,
        fullpath=fullpath,
    )

@app.route('/show/<filename>')
def uploaded_file(filename):
    return render_template('go.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)