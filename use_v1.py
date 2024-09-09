import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Đường dẫn tới file ảnh CAPTCHA bạn đã tải lên
img_path =f'test/hhhgh.jpg'

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('captcha_model.h5')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
valid_chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
le.fit(valid_chars)

# Hàm dự đoán CAPTCHA từ ảnh
def predict_captcha(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_class = [np.argmax(p, axis=1)[0] for p in predictions]
    predicted_label = le.inverse_transform(predicted_class)
    return predicted_label

predicted_captcha = predict_captcha(img_path)
print("Mã CAPTCHA dự đoán:", ''.join(predicted_captcha))
