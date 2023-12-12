from flask import Flask , request, jsonify # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api  # Api 구현을 위한 Api 객체 import
from predict.Clothe import Predictor as ClothePredictor
from predict.Label import Predictor as LabelPredictor
from predict.Stain import Predictor as StainPredictor
from predict.Pattern import Predictor as PatternPredictor

import json


app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

clothe_predictor = ClothePredictor()
label_predictor = LabelPredictor()
stain_predictor = StainPredictor()
pattern_predictor = PatternPredictor()

@app.route('/clothes', methods=['POST'])
def clothe_post():
    # Retrieve the image data from the request body
    image = request.files['image'] 

    dict = clothe_predictor.predict(image)

    # Return a response if required
    return jsonify(dict)

@app.route('/stains', methods=['POST'])
def stain_post():
    # Retrieve the image data from the request body
    image = request.files['image'] 

    dict=stain_predictor.predict(image)

    # Return a response if required
    return jsonify(dict)

@app.route('/labels', methods=['POST'])
def label_post():
    # Retrieve the image data from the request body
    print(request.content_type)

    print(request.args.getlist)

    image_data1 = request.files['image']
    # image_data2 = request.form['image']

    print(image_data1.content_type)
    # print(image_data2.content_type)

    dict=label_predictor.predict(image_data1)

    # Return a response if required
    return str(dict)

@app.route('/patterns', methods=['POST'])
def pattern_post():
    # Retrieve the image data from the request body
    image = request.files['image']
    print(image.content_type)

    dict=pattern_predictor.predict(image)

    # Return a response if required
    return jsonify(dict)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=10002)