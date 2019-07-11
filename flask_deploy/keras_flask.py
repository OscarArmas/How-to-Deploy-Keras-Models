# Import flask which will handle our communication with the frontend
# Also import few other libraries
from flask import Flask, render_template, request
import imageio
import numpy as np
import re
import sys
import base64
import os
from load import *
import cv2
# Path to our saved model
sys.path.append(os.path.abspath("./model"))

# Initialize flask app
app = Flask(__name__)
#Initialize some global variables
global model, graph
model, graph = init()
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
 # Predict method is called when we push the 'Predict' button
 # on the webpage. We will feed the user drawn image to the model
 # perform inference, and return the classification
    imgData = request.get_data()
    convertImage(imgData)
 # read the image into memory
    x = cv2.imread('output.png',0)
    x = np.invert(x)
    print(type(x))
 # make it the right size
    x = cv2.resize(x,dsize=(28,28))
    print(x.shape)
 #You can save the image (optional) to view later
    imageio.imwrite('final_image.png', x)
    x = x.reshape(1, 28, 28, 1)

    with graph.as_default():
        out = model.predict(x)
        response = np.argmax(out, axis=1)
    return str(response[0])

if __name__ == "__main__":
# run the app locally on the given port
    app.run(host='localhost', port=5000)

