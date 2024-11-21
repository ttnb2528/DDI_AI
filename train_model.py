import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision
from preprocess_data import load_processed_data

# Cấu hình GPU
def setup_gpu():
    """
    Cấu hình và tối ưu GPU
    """
    try:
        # Liệt kê các GPU có sẵn
        gpus = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
        print("GPU Info:", gpus)
        
        if gpus:
            for gpu in gpus:
                # Cho phép tăng bộ nhớ động
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Cấu hình để sử dụng GPU hiệu quả
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)]  # Giới hạn 4GB
            )
            
            # Cấu hình mixed precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print('Mixed precision policy:', policy)
            
            # Cấu hình XLA
            tf.config.optimizer.set_jit(True)
            
            print("GPU đã được cấu hình thành công!")
    except Exception as e:
        print(f"Lỗi khi cấu hình GPU: {str(e)}")

# Gọi hàm setup_gpu() ở đầu script
setup_gpu()

def extract_drug_features(row, drug_data):
    """
    Trích xuất đặc trưng cho một cặp thuốc
    """
    try:
        drug_info_df = drug_data['/kaggle/working/DDI_AI/Supplemental_Table_S2.xlsx']
        ddi_df = drug_data['/kaggle/working/DDI_AI/Supplemental_Table_S3.xlsx']
        
        # Trích xuất features cho từng thuốc
        drug1_features = extract_single_drug_features(
            drug_info_df, 
            row['Drug1\nDrugBank accession'], 
            ddi_df
        )
        
        drug2_features = extract_single_drug_features(
            drug_info_df, 
            row['Drug2\nDrugBank accession'], 
            ddi_df
        )
        
        # Thêm features tương tác
        interaction_features = np.array([
            float(row['Score 1']) if pd.notna(row['Score 1']) else 0,
            1 if row['DDI in\nDrugBank'] == 'Yes' else 0,
            1 if row['Negative\nhealth effect'] else 0,
            len(row['Sentence'].split()) if pd.notna(row['Sentence']) else 0
        ])
        
        # Kết hợp tất cả features
        combined_features = np.concatenate([drug1_features, drug2_features, interaction_features])
        return combined_features
        
    except Exception as e:
        print(f"Lỗi khi trích xuất features: {str(e)}")
        return None

def extract_single_drug_features(drug_info_df, drug_id, ddi_df):
    """
    Trích xuất đặc trưng cho một thuốc
    """
    drug_row = drug_info_df[drug_info_df['DrugBank\naccession'] == drug_id]
    if len(drug_row) == 0:
        return np.zeros(48)
    
    features = []
    
    # 1. Thông tin cơ bản về thuốc
    features.append(1 if drug_row['Small\nmolecule'].iloc[0] == 'Yes' else 0)
    features.append(1 if drug_row['DMD\nfor MS'].iloc[0] == 'Yes' else 0)
    
    # 2. Thông tin về bệnh nhân
    patients = drug_row['Number of\npatients 2'].iloc[0]
    features.append(float(patients) if pd.notna(patients) else 0)
    
    # 3. Phân tích tương tác
    drug_as_drug1 = ddi_df[ddi_df['Drug1\nDrugBank accession'] == drug_id]
    drug_as_drug2 = ddi_df[ddi_df['Drug2\nDrugBank accession'] == drug_id]
    
    # 3.1 Thống kê cho từng vai trò
    for drug_interactions in [drug_as_drug1, drug_as_drug2]:
        if len(drug_interactions) > 0:
            features.extend([
                len(drug_interactions),  # Số lượng tương tác
                drug_interactions['Negative\nhealth effect'].mean(),  # Tỷ lệ ảnh hưởng tiêu cực
                drug_interactions['Score 1'].mean() if pd.notna(drug_interactions['Score 1']).any() else 0,  # Điểm trung bình
                drug_interactions['DDI in\nDrugBank'].eq('Yes').mean(),  # Tỷ lệ có trong DrugBank
                len(drug_interactions['DDI type'].unique())  # Số loại tương tác khác nhau
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
    
    # 3.2 Phân tích các loại tương tác phổ biến
    all_interactions = pd.concat([drug_as_drug1, drug_as_drug2])
    if len(all_interactions) > 0:
        ddi_type_counts = all_interactions['DDI type'].value_counts(normalize=True)
        common_types = ['DDI type 15', 'DDI type 24', 'DDI type 26', 'DDI type 30']
        for ddi_type in common_types:
            features.append(ddi_type_counts.get(ddi_type, 0))
    else:
        features.extend([0] * len(common_types))
    
    # Pad vector to length 48
    features.extend([0] * (48 - len(features)))
    return np.array(features)

def create_model(input_shape, num_classes):
    """
    Tạo mô hình neural network với tối ưu cho GPU
    """
    # Xóa session cũ
    tf.keras.backend.clear_session()
    
    model = Sequential([
        Dense(2048, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Sử dụng optimizer với mixed precision
    optimizer = Adam(learning_rate=0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # Sử dụng XLA compilation
    )
    
    return model

def train_model():
    # Load dữ liệu đã xử lý
    processed_data = load_processed_data()
    X = processed_data['X']
    y = processed_data['y']
    num_classes = processed_data['num_classes']
    ddi_type_to_index = processed_data['ddi_type_to_index']
    scaler = processed_data['scaler']
    
    print(f"\nThông tin dữ liệu:")
    print(f"Số lượng mẫu: {len(X)}")
    print(f"Số lượng lớp: {num_classes}")
    print(f"Kích thước features: {X.shape[1]}")
    
    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Chuẩn hóa dữ liệu
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Tạo và huấn luyện mô hình
    print("\nĐang tạo mô hình...")
    model = create_model(input_shape=(X_train.shape[1],), num_classes=num_classes)
    
    print("\nTóm tắt mô hình:")
    model.summary()
    
    print("\nBắt đầu huấn luyện...")
    # Thêm callbacks để tối ưu quá trình huấn luyện
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        # Thêm TensorBoard callback để theo dõi quá trình huấn luyện
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Tăng batch_size khi sử dụng GPU
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,  # Tăng batch size để tận dụng GPU tốt hơn
        callbacks=callbacks,
        verbose=1
    )
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình trên tập test:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Lưu mô hình và scaler
    model.save('trained_ddi_model.keras')
    np.save('ddi_type_mapping.npy', ddi_type_to_index)
    print("\nĐã lưu mô hình tại 'trained_ddi_model.keras'")
    print("Đã lưu mapping tại 'ddi_type_mapping.npy'")
    
    return model, history, scaler, ddi_type_to_index

if __name__ == "__main__":
    model, history, scaler, ddi_type_to_index = train_model() 