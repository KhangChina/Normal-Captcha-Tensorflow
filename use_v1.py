import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Đường dẫn tới file ảnh CAPTCHA bạn đã tải lên
img_path = 'D:/python/Normal Captcha/test/2024-09-09_15h54_01.jpg'

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('captcha_model.h5')

# Chuyển đổi nhãn (giả sử bạn đã sử dụng LabelEncoder với các ký tự 0-9 và A-Z)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
valid_chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
le.fit(valid_chars)

# Hàm dự đoán CAPTCHA từ ảnh
def predict_captcha(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Điều chỉnh kích thước ảnh nếu cần
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

    # Chuẩn hóa dữ liệu
    img_array /= 255.0

    # Dự đoán
    predictions = model.predict(img_array)
    
    # Áp dụng np.argmax cho từng ký tự dự đoán
    predicted_class = [np.argmax(p, axis=1)[0] for p in predictions]

    # Chuyển đổi thành ký tự tương ứng
    predicted_label = le.inverse_transform(predicted_class)
    return predicted_label

# Gọi hàm dự đoán
predicted_captcha = predict_captcha(img_path)
print("Mã CAPTCHA dự đoán:", ''.join(predicted_captcha))
