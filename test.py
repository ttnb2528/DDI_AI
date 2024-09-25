import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  # For data augmentation

# Thiết lập môi trường
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Hàm đọc dữ liệu
def read_data():
    tables = {
        'S1': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S1.xlsx', sheet_name='Supplemental Table 1'),
        'S2': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S2.xlsx', sheet_name='Supplemental Table 2'),
        'S3': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S3.xlsx', sheet_name='Supplemental Table 3'),
        'S4': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S4.xlsx', sheet_name='Supplemental Table 4'),
    }
    return tables

# Hàm tiền xử lý dữ liệu
def preprocess_data(tables):
    table_s3 = tables['S3'].copy()
    table_s2 = tables['S2'][['DrugBank\naccession', 'Number of\npatients 2']].astype(str)
    
    # Bổ sung số lượng bệnh nhân
    table_s3 = table_s3.merge(table_s2, left_on='Drug1\nDrugBank accession', right_on='DrugBank\naccession', how='left')
    table_s3 = table_s3.merge(table_s2, left_on='Drug2\nDrugBank accession', right_on='DrugBank\naccession', suffixes=('_Drug1', '_Drug2'), how='left')
    
    # Điền giá trị thiếu nếu có
    table_s3.fillna(0, inplace=True)

    # Tạo các đặc trưng và nhãn
    X = table_s3[['Drug1\nDrugBank accession', 'Drug2\nDrugBank accession', 'DDI type', 'Number of\npatients 2', 'Alternative approved\ndrugs for Drug1 3', 'Alternative approved\ndrugs for Drug2 3']]
    y = table_s3['Negative\nhealth effect']

    # Chuyển đổi dữ liệu thành số
    X = pd.get_dummies(X, columns=['Drug1\nDrugBank accession', 'Drug2\nDrugBank accession'], drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, y

# Hàm xây dựng mô hình
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Thay đổi optimizer và learning rate nếu cần
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Hàm huấn luyện mô hình và đánh giá
def train_and_evaluate(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_scores = []
    best_model = None

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]  # X là numpy array
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Xử lý dữ liệu mất cân bằng
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = build_model(X_train_res.shape[1])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Huấn luyện mô hình
        history = model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        # Đánh giá mô hình
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Fold accuracy: {accuracy}, Fold loss: {loss}')
        all_scores.append(accuracy)

        # Dự đoán và tính các chỉ số khác
        y_pred_classes = (model.predict(X_test) > 0.5).astype("int32")
        precision = precision_score(y_test, y_pred_classes)
        recall = recall_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes)
        print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

        # Lưu lại mô hình tốt nhất
        if best_model is None or accuracy > max(all_scores):
            best_model = model

    # In ra độ chính xác trung bình
    print(f'Average accuracy from KFold: {sum(all_scores) / len(all_scores)}')

    # Lưu mô hình tốt nhất
    best_model.save('my_deepddi_model_kfold_smote.h5')

# Chạy chương trình
tables = read_data()
X, y = preprocess_data(tables)
X = StandardScaler().fit_transform(X)  # Chuẩn hóa dữ liệu

train_and_evaluate(X, y)
