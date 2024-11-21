import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import time

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

class ModelComparison:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.results = {}
    
    def train_random_forest(self):
        """Huấn luyện và tối ưu Random Forest"""
        print("\nHuấn luyện Random Forest...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
        
        start_time = time.time()
        grid_search.fit(self.X_train_scaled, self.y_train)
        training_time = time.time() - start_time
        
        y_pred = grid_search.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.results['random_forest'] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"Random Forest Best Accuracy: {accuracy:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")
    
    def train_xgboost(self):
        """Huấn luyện và tối ưu XGBoost"""
        print("\nHuấn luyện XGBoost...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb = XGBClassifier(random_state=42)
        grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1, verbose=1)
        
        start_time = time.time()
        grid_search.fit(self.X_train_scaled, self.y_train)
        training_time = time.time() - start_time
        
        y_pred = grid_search.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.results['xgboost'] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"XGBoost Best Accuracy: {accuracy:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

def main():
    # Điều chỉnh đường dẫn file
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
    
    print(f"Tổng số mẫu dữ liệu: {len(ddi_df)}")
    for idx in range(len(ddi_df)):
        if idx % 1000 == 0:
            print(f"Đang xử lý mẫu thứ {idx}...")
        
        row = ddi_df.iloc[idx]
        features = extract_drug_features(row, drug_data)
        if features is not None:
            features_list.append(features)
            ddi_type = int(row['DDI type'].split()[-1])
            labels.append(ddi_type)
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Khởi tạo và chạy so sánh
    comparison = ModelComparison(X, y)
    
    # Huấn luyện các mô hình
    comparison.train_random_forest()
    comparison.train_xgboost()
    
    # Vẽ biểu đồ so sánh
    comparison.plot_results()
    
    # Tạo báo cáo
    report = comparison.generate_report()
    print(report)

if __name__ == "__main__":
    main() 