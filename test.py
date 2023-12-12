from predict.Pattern import Predictor
import cv2
import io
from PIL import Image

clothe_predictor = Predictor()

img = Image.open('./0.png', mode='r')

buffer = io.BytesIO()
img.save(buffer, format=img.format)
byte_array = buffer.getvalue()

print(clothe_predictor.predict(byte_array))