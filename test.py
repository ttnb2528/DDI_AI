import os

# Thiết lập biến môi trường để tắt các tùy chọn oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ các bảng Excel
table_s1 = pd.read_excel('Supplemental_Table_S1.xlsx', sheet_name='Supplemental Table 1')
table_s2 = pd.read_excel('Supplemental_Table_S2.xlsx', sheet_name='Supplemental Table 2')
table_s3 = pd.read_excel('Supplemental_Table_S3.xlsx', sheet_name='Supplemental Table 3')
table_s4 = pd.read_excel('Supplemental_Table_S4.xlsx', sheet_name='Supplemental Table 4')

# Hiển thị thông tin dữ liệu
print('Supplemental Table 1')
print(table_s1.head())
print('Supplemental Table 2')
print(table_s2.head())
print('Supplemental Table 3')
print(table_s3.head())
print('Supplemental Table 4')
print(table_s4.head())

# Chuyển đổi các dữ liệu categorical thành dạng số
le = LabelEncoder()
table_s3['Drug1\nDrugBank accession'] = le.fit_transform(table_s3['Drug1\nDrugBank accession'])
table_s3['Drug2\nDrugBank accession'] = le.fit_transform(table_s3['Drug2\nDrugBank accession'])
table_s3['DDI type'] = le.fit_transform(table_s3['DDI type'])
table_s3['Negative\nhealth effect'] = le.fit_transform(table_s3['Negative\nhealth effect'])

# Tạo các đặc trưng và nhãn từ dữ liệu
X = table_s3[['Drug1\nDrugBank accession', 'Drug2\nDrugBank accession', 'DDI type']]
y = table_s3['Negative\nhealth effect']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển đổi kiểu dữ liệu thành float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Xây dựng mô hình
input_dim = X_train.shape[1]  # Số đặc trưng đầu vào
num_classes = len(set(y))  # Số lớp đầu ra

model = Sequential([
    Dense(2048, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2048, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2048, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

# Tính accuracy
print('Accuracy:', accuracy_score(y_test, y_pred_classes))

# Lưu mô hình
model.save('my_deepddi_model.h5')

# Tải mô hình
model = tf.keras.models.load_model('my_deepddi_model.h5')
