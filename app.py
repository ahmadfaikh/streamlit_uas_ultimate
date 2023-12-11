# from scipy.stats import skew, kurtosis, mode, iqr
# from streamlit_option_menu import option_menu
# from sklearn.model_selection import GridSearchCV
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import os
# import seaborn as sns
# import scipy.stats
# import librosa
import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier  # For classification tasks
from sklearn.ensemble import RandomForestRegressor   # For regression tasks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pickle
from pickle import dump
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import library yang dibutuhkan untuk melakukan data preprocessing

st.title("Glass Identification")
RI = st.number_input("RI", min_value=0.0, max_value=100.0, step=0.1)
Na2O = st.number_input("Na2O", min_value=0.0, max_value=100.0, step=0.1)
Mg = st.number_input("Mg", min_value=0.0, max_value=100.0, step=0.1)
Al2O3 = st.number_input("Al2O3", min_value=0.0,
                        max_value=100.0, step=0.1)
SIO2 = st.number_input("SIO2", min_value=0.0, max_value=100.0, step=0.1)
K2O = st.number_input("K2O", min_value=0.0, max_value=100.0, step=0.1)
CAO = st.number_input("CAO", min_value=0.0, max_value=100.0, step=0.1)
BAO = st.number_input("BAO", min_value=0.0, max_value=100.0, step=0.1)
Fe2O3 = st.number_input("Fe2O3", min_value=0.0,
                        max_value=100.0, step=0.1)
results = []
result = {
    'RI': float(RI),
    'na2o': float(Na2O),
    'mg': float(Mg),
    'al2o3': float(Al2O3),
    'sio2': float(SIO2),
    'k2o': float(K2O),
    'cao': float(CAO),
    'bao': float(BAO),
    'fe2o3': float(Fe2O3)
}
results.append(result)
print(result)
data_tes = pd.DataFrame(results)
# load model KNN dan PCA terbaik
# load_pca = pickle.load(open('best_pca_component.pkl', 'rb'))
scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))

df = pd.read_csv('glass_dataset_smote.csv')
df = df.iloc[:, 1:]
# Memisahkan kolom target (label) dari kolom fitur
X = df.drop(columns=['label'])  # Kolom fitur
y = df['label']  # Kolom target

# Normalisasi data menggunakan StandardScaler

X_scaled = scaler.fit_transform(X)

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Access hyperparameters
# best_n_neighbors =

# Melakukan PCA pada data audio yang diunggah
# pca = PCA(n_components=load_pca['best_component'])

# Memanggil metode fit dengan data pelatihan sebelum menggunakan transform
X_test = scaler.transform(data_tes)

# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# Membuat model KNN dengan hyperparameter terbaik
best_knn_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_knn_model.fit(X_train, y_train)

predicted_label = best_knn_model.predict(X_test)
if st.button("Prediksi"):
    if predicted_label == 1:
        st.write("Label Predict:", '-- 1 building_windows_float_processed')
    if predicted_label == 2:
        st.write("Label Predict:", '-- 2 building_windows_non_float_processed')
    if predicted_label == 3:
        st.write("Label Predict:", '-- 3 vehicle_windows_float_processed')
    if predicted_label == 4:
        st.write("Label Predict:", '-- 4 vehicle_windows_non_float_processed')
    if predicted_label == 5:
        st.write("Label Predict:", '-- 5 containers')
    if predicted_label == 6:
        st.write("Label Predict:", '-- 6 tableware')
    if predicted_label == 7:
        st.write("Label Predict:", '-- 7 headlamps')
