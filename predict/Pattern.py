import tensorflow as tf
import pandas as pd
import json
import io
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self):
        # Load the trained model
        self.model = load_model('predict/model/pattern.h5')
        print("Loading success from predict/model/pattern.h5")
        # self.class_dictionary = {'animal': 0,
        #                          'check': 1,
        #                          'chevron': 2,
        #                          'diamond': 3,
        #                          'floral': 4,
        #                          'graphic': 5,
        #                          'lettering': 6,
        #                          'plain': 7,
        #                          'polka dots': 8,
        #                          'stripes': 9}
        
        self.IMAGE_SIZE = (244, 244)

    def predict(self, file):
        byte = file.read()
        image_pil = Image.open(io.BytesIO(byte))
        image_resized = image_pil.resize(self.IMAGE_SIZE)
        test_image = image.img_to_array(image_resized)

        test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
        test_image = preprocess_input(test_image)
        prediction = self.model.predict(test_image)

        df = pd.DataFrame({'pred': prediction[0]})
        df = df.sort_values(by='pred', ascending=False, na_position='first')

        score = (df.iloc[0]['pred'])* 100

        pattern = 10

        if score >= 60:
            pattern = df[df == df.iloc[0]].index[0]
            print(pattern)

        dict = {'pattern': int(pattern)}

        return dict