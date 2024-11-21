import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import randint
from joblib import dump, load
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import ADASYN

# Thiết lập môi trường
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Hàm đọc dữ liệu từ tệp Excel
def read_data():
    return {
        'S1': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S1.xlsx'),
        'S2': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S2.xlsx'),
        'S3': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S3.xlsx'),
        'S4': pd.read_excel('/kaggle/working/DDI_AI/Supplemental_Table_S4.xlsx')
    }

# Hàm tiền xử lý dữ liệu
def preprocess_data(tables):
    table_s3 = tables['S3']
    table_s2 = tables['S2'].astype(str)

    # Ghép các bảng
    table_s3 = table_s3.merge(table_s2, left_on='Drug1\nDrugBank accession', right_on='DrugBank\naccession', how='left')
    table_s3 = table_s3.merge(table_s2, left_on='Drug2\nDrugBank accession', right_on='DrugBank\naccession', suffixes=('_Drug1', '_Drug2'), how='left')

    # Xử lý dữ liệu thiếu
    table_s3.fillna(0, inplace=True)

    # Lựa chọn cột
    X = table_s3[['Drug1\nDrugBank accession', 'Drug2\nDrugBank accession', 'DDI type', 'Number of\npatients 2', 'Alternative approved\ndrugs for Drug1 3', 'Alternative approved\ndrugs for Drug2 3']]
    y = table_s3['Negative\nhealth effect']

    # Chuyển thành one-hot encoding và xử lý dữ liệu thiếu
    X = pd.get_dummies(X, columns=['Drug1\nDrugBank accession', 'Drug2\nDrugBank accession'], drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Sử dụng SimpleImputer thay vì KNNImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y

# Huấn luyện mô hình
# def train_model(X, y):
#     param_dist = {
#         'clf__n_estimators': randint(50, 200),
#         'clf__max_depth': [None, 10, 20, 30],
#         'clf__min_samples_split': randint(2, 10),
#         'clf__min_samples_leaf': randint(1, 5)
#     }

#     # Áp dụng SMOTETomek để xử lý mất cân bằng
#     smote_tomek = SMOTETomek(random_state=42)
#     X_res, y_res = smote_tomek.fit_resample(X, y)

#     # Xây dựng pipeline
#     pipeline = Pipeline([ 
#         ('scaler', StandardScaler()),  # Chuẩn hóa
#         ('constant_filter', VarianceThreshold(threshold=0)),  # Loại bỏ các đặc trưng không thay đổi
#         ('selector', SelectKBest(score_func=f_classif, k=10)),  # Chọn 10 đặc trưng tốt nhất
#         ('pca', PCA()),  # Giảm chiều dữ liệu
#         ('clf', RandomForestClassifier(random_state=42))  # RandomForest
#     ])

#     # Sử dụng Stratified K-Fold
#     stratified_kf = StratifiedKFold(n_splits=3)

#     # Tìm tham số tốt nhất với RandomizedSearchCV
#     random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=stratified_kf, scoring='accuracy', verbose=2, n_jobs=-1, random_state=42)

#     random_search.fit(X_res, y_res)

#     # Lưu lại pipeline tốt nhất
#     dump(random_search.best_estimator_, 'final_pipeline_with_best_params.joblib')

#     return random_search.best_estimator_

def train_model(X, y):
    param_dist = {
        'clf__n_estimators': randint(100, 500),
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': randint(2, 10),
        'clf__min_samples_leaf': randint(1, 5)
    }

    # Sử dụng ADASYN thay cho SMOTETomek để xử lý mất cân bằng
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X, y)

    # Xây dựng pipeline
    pipeline = Pipeline([ 
        ('scaler', StandardScaler()),  
        ('constant_filter', VarianceThreshold(threshold=0)),  
        ('selector', RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=10)),  
        ('clf', GradientBoostingClassifier(random_state=42))  
    ])

    # Stratified K-Fold với 5 splits thay vì 3
    stratified_kf = StratifiedKFold(n_splits=5)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=stratified_kf, scoring='accuracy', verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X_res, y_res)

    # Lưu lại pipeline tốt nhất
    dump(random_search.best_estimator_, 'final_pipeline_with_best_params.joblib')

    return random_search.best_estimator_

# Dự đoán
def predict_with_model(X_new, model_file='final_pipeline_with_best_params.joblib'):
    model = load(model_file)
    predictions = model.predict(X_new)
    return predictions

# ======== Quy trình chính ========
if __name__ == '__main__':
    # Đọc và xử lý dữ liệu
    tables = read_data()
    X, y = preprocess_data(tables)

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện và lưu mô hình
    model = train_model(X_train, y_train)

    # Dự đoán toàn bộ tập kiểm thử
    predictions = predict_with_model(X_test)

    # Tính toán tỷ lệ chính xác
    accuracy = accuracy_score(y_test, predictions)
    print("Tỷ lệ chính xác:", accuracy)
