import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Hàm load ảnh từ thư mục và tạo dữ liệu đầu vào, nhãn
def load_data_from_folder(folder_path, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            images.append(img_array)
            label = filename.split('.')[0]  # Lấy tên tệp (ví dụ 11438)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load dữ liệu
X_train, y_train = load_data_from_folder('train')
X_test, y_test = load_data_from_folder('test1')

# Chuẩn hóa ảnh
X_train = X_train / 255.0
X_test = X_test / 255.0

# Chuyển đổi nhãn thành các ký tự riêng lẻ và mã hóa
def encode_labels(labels):
    encoded_labels = []
    for label in labels:
        encoded_label = []
        for char in label:
            encoded_label.append(le.transform([char])[0])
        encoded_labels.append(encoded_label)
    return np.array(encoded_labels)

# Khởi tạo LabelEncoder
valid_chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
le = LabelEncoder()
le.fit(valid_chars)

y_train = encode_labels(y_train)
y_test = encode_labels(y_test)

# Xây dựng mô hình
def build_model(input_shape, num_classes, num_chars):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # Tạo 5 đầu ra, mỗi đầu ra tương ứng với một ký tự
    outputs = []
    for _ in range(num_chars):
        outputs.append(layers.Dense(num_classes, activation='softmax')(x))

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'] * num_chars)
    return model

input_shape = (64, 64, 3)  # Kích thước ảnh (64x64 pixels, 3 kênh màu)
num_classes = len(valid_chars)  # 36 lớp (10 chữ số + 26 chữ cái)
num_chars = 5  # Mỗi CAPTCHA có 5 ký tự

model = build_model(input_shape, num_classes, num_chars)
model.summary()

# Huấn luyện mô hình
y_train_split = [y_train[:, i] for i in range(num_chars)]
y_test_split = [y_test[:, i] for i in range(num_chars)]

history = model.fit(X_train, y_train_split, epochs=100, validation_data=(X_test, y_test_split))

# Lưu mô hình
model.save('captcha_model.h5')
print("Train success")

