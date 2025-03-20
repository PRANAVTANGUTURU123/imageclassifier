from PIL import Image
from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
model = None
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_classifier_model():
    try:
        return load_model('model1.h5')
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

@app.before_request
def before_request():
    global model
    if model is None:
        model = load_classifier_model()

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', error="No file part")

    imagefile = request.files['imagefile']

    if imagefile.filename == '':
        return render_template('index.html', error="No selected file")

    if imagefile and allowed_file(imagefile.filename):
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        img = Image.open(image_path)
        img = img.convert("RGB")  # Convert to RGB to ensure it has 3 channels
        img = img.resize((32, 32))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]

        classification = '%s' % (predicted_class_name)

        return render_template('index.html', prediction=classification, image_path=image_path)

    else:
        return render_template('index.html', error="Invalid file type. Allowed types are .png, .jpg, .jpeg")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(port=5678, debug=True)
 