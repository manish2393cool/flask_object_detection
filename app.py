# from ObjectDetector import Detector
import io

from flask import Flask, render_template, request
from PIL import Image
from flask import send_file
import pickle
"""------------------------------------------------"""
app = Flask(__name__)
model = pickle.load(open('ae_model.pkl', 'rb'))

# detector = Detector()


"""------------------Index page------------------------------"""
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def upload():
    if request.method == 'POST':
        file = Image.open(request.files['file'].stream)
        # img = detector.detectObject(file)
        print(file)
        img = model.predict(file)
        return send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')


if __name__ == "__main__":
    app.run()