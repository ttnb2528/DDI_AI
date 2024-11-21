import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

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

def preprocess_and_save_data():
    """
    Xử lý dữ liệu một lần và lưu vào file
    """
    # Điều chỉnh đường dẫn cho Kaggle
    save_path = '/kaggle/working/processed_data.pkl'
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
    
    # Lưu với đường dẫn mới
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nĐã lưu dữ liệu đã xử lý vào '{save_path}'")
    return processed_data

def load_processed_data():
    """
    Load dữ liệu đã xử lý từ file
    """
    # Điều chỉnh đường dẫn cho Kaggle
    save_path = '/kaggle/working/processed_data.pkl'
    try:
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        print("Đã load dữ liệu đã xử lý từ file")
        return data
    except FileNotFoundError:
        print("Chưa tìm thấy file dữ liệu đã xử lý. Tiến hành xử lý mới...")
        return preprocess_and_save_data()

if __name__ == "__main__":
    preprocess_and_save_data() 