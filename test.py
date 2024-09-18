import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l2

# Đọc dữ liệu từ các bảng Excel
table_s1 = pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S1.xlsx', sheet_name='Supplemental Table 1')
table_s2 = pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S2.xlsx', sheet_name='Supplemental Table 2')
table_s3 = pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S3.xlsx', sheet_name='Supplemental Table 3')
table_s4 = pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S4.xlsx', sheet_name='Supplemental Table 4')

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

# Xây dựng mô hình với điều chỉnh để giảm overfitting
input_dim = X_train.shape[1]  # Số đặc trưng đầu vào
num_classes = len(set(y))  # Số lớp đầu ra

model = Sequential([
    Input(shape=(input_dim,)),  
    Dense(256, activation='relu', kernel_regularizer=l2(0.02)),  # Giảm số lượng neurons, tăng l2
    BatchNormalization(),
    Dropout(0.6),  # Tăng Dropout
    Dense(128, activation='relu', kernel_regularizer=l2(0.02)),  # Giảm số neurons
    BatchNormalization(),
    Dropout(0.6),  # Tăng Dropout
    Dense(64, activation='relu', kernel_regularizer=l2(0.02)),  # Giảm số neurons
    BatchNormalization(),
    Dropout(0.6),  # Tăng Dropout
    Dense(num_classes, activation='softmax')  # Output layer với số lớp
])

# Compile model với learning rate nhỏ
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Thêm EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huấn luyện mô hình với EarlyStopping
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

# Tính accuracy
print('Accuracy:', accuracy_score(y_test, y_pred_classes))

# Lưu mô hình
model.save('my_deepddi_model.h5')
