import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def preprocess_and_save_data():
    """
    Xử lý dữ liệu một lần và lưu vào file
    """
    excel_files = [
        '/kaggle/working/DDI_AI/Supplemental_Table_S1.xlsx',
        '/kaggle/working/DDI_AI/Supplemental_Table_S2.xlsx',
        '/kaggle/working/DDI_AI/Supplemental_Table_S3.xlsx',
        '/kaggle/working/DDI_AI/Supplemental_Table_S4.xlsx'
    ]
    
    print("Đang tải và xử lý dữ liệu...")
    drug_data = {}
    for file_path in excel_files:
        drug_data[file_path] = pd.read_excel(file_path)
    
    # Xử lý dữ liệu và tạo features
    ddi_df = drug_data[excel_files[2]]  # Table S3
    features_list = []
    labels = []
    
    # Tạo mapping cho các DDI types
    unique_ddi_types = sorted(ddi_df['DDI type'].unique())
    ddi_type_to_index = {ddi_type: idx for idx, ddi_type in enumerate(unique_ddi_types)}
    
    print(f"Tổng số mẫu dữ liệu: {len(ddi_df)}")
    for idx in range(len(ddi_df)):
        if idx % 1000 == 0:
            print(f"Đang xử lý mẫu thứ {idx}...")
        
        row = ddi_df.iloc[idx]
        features = extract_drug_features(row, drug_data)
        if features is not None:
            features_list.append(features)
            ddi_type = row['DDI type']
            label_idx = ddi_type_to_index[ddi_type]
            labels.append(label_idx)
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lưu dữ liệu đã xử lý
    processed_data = {
        'X': X_scaled,
        'y': y,
        'ddi_type_to_index': ddi_type_to_index,
        'scaler': scaler,
        'num_classes': len(unique_ddi_types)
    }
    
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\nĐã lưu dữ liệu đã xử lý vào 'processed_data.pkl'")
    return processed_data

def load_processed_data():
    """
    Load dữ liệu đã xử lý từ file
    """
    try:
        with open('processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Đã load dữ liệu đã xử lý từ file")
        return data
    except FileNotFoundError:
        print("Chưa tìm thấy file dữ liệu đã xử lý. Tiến hành xử lý mới...")
        return preprocess_and_save_data()

if __name__ == "__main__":
    preprocess_and_save_data() 