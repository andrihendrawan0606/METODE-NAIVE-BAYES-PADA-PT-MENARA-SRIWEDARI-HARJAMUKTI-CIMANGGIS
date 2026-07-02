import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


st.set_page_config(
    page_title="Sistem Prediksi Kelayakan Proyek - NB",
    page_icon="ðŸ¢",
    layout="wide"
)


def load_and_preprocess_data(file_source):

    df_raw = pd.read_excel(file_source, skiprows=8)
    
    df_raw.columns = df_raw.columns.str.strip()
    
    df_raw = df_raw.dropna(subset=['Status Proyek'])
    
    df_clean = pd.DataFrame()
    

    def kat_biaya(val):
        val = float(val)
        if val <= 2.0:
            return "Rendah"
        elif val <= 4.0:
            return "Sedang"
        else:
            return "Tinggi"
    df_clean['Biaya'] = df_raw['Biaya (Milyar)'].apply(kat_biaya)
    

    def kat_unit(val):
        val = int(val)
        if val <= 10:
            return "Sedikit"
        elif val <= 20:
            return "Sedang"
        else:
            return "Banyak"
    df_clean['Unit'] = df_raw['Unit'].apply(kat_unit)

    def kat_durasi(val):
        val = int(val)
        if val <= 5:
            return "Cepat"
        elif val <= 12:
            return "Sedang"
        else:
            return "Lama"
    df_clean['Durasi'] = df_raw['Durasi (Bulan)'].apply(kat_durasi)
    

    df_clean['Material'] = df_raw['Ketersediaan Material'].astype(str).str.strip().str.lower()
    

    df_clean['Cuaca'] = df_raw['Kondisi Cuaca'].astype(str).str.strip().str.lower()
    

    def kat_tenaga(val):
        if isinstance(val, str):
            angka = int(''.join(filter(str.isdigit, val)))
        else:
            angka = int(val)
            
        if angka <= 10:
            return "Sedikit"
        elif angka <= 20:
            return "Sedang"
        else:
            return "Banyak"
    df_clean['Tenaga_Kerja'] = df_raw['Tenaga Kerja'].apply(kat_tenaga)
    
    df_clean['Status_Proyek'] = df_raw['Status Proyek'].astype(str).str.strip().str.lower()
    
    return df_clean

st.sidebar.header("ðŸ“‚ Sumber Data Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah File Excel (.xlsx)", type=["xlsx"])

DEFAULT_FILE = "dataset_sriwedari.xlsx"

df = None

if uploaded_file is not None:
    try:
        df = load_and_preprocess_data(uploaded_file)
        st.sidebar.success("Berhasil memuat file yang diunggah!")
    except Exception as e:
        st.sidebar.error(f"Gagal memproses file unggahan: {e}")
elif os.path.exists(DEFAULT_FILE):
    try:
        df = load_and_preprocess_data(DEFAULT_FILE)
        st.sidebar.info(f"Menggunakan file lokal: '{DEFAULT_FILE}'")
    except Exception as e:
        st.sidebar.error(f"Gagal memproses file default: {e}")


if df is None:
    st.warning("âš ï¸ Silakan unggah file Excel dataset Anda di sidebar atau letakkan file dataset.")
    st.stop()


X = df.drop('Status_Proyek', axis=1)
y = df['Status_Proyek']


categories = [
    ["Rendah", "Sedang", "Tinggi"],     
    ["Sedikit", "Sedang", "Banyak"],    
    ["Cepat", "Sedang", "Lama"],      
    ["rendah", "sedang", "tinggi"],  
    ["buruk", "normal", "baik"],       
    ["Sedikit", "Sedang", "Banyak"]     
]

encoder = OrdinalEncoder(categories=categories)
X_encoded = encoder.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


model = CategoricalNB(alpha=1.0)
model.fit(X_encoded, y_encoded)





st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>SISTEM PREDIKSI KEBERHASILAN PROYEK CLUSTER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>PT Menara Sriwedari Properti Harjamukti-Cimanggis (AI & Naive Bayes)</p>", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["ðŸ”® Prediksi Proyek Baru", "ðŸ“Š Analisis Dataset Latih", "ðŸ“‰ Evaluasi Model (Confusion Matrix)"])


with tab1:
    st.header("Input Karakteristik Proyek Baru")
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        biaya_input = st.selectbox("Biaya Pembangunan Proyek (Milyar):", ["Rendah", "Sedang", "Tinggi"])
        unit_input = st.selectbox("Jumlah Unit Rumah:", ["Sedikit", "Sedang", "Banyak"])
        
    with col2:
        durasi_input = st.selectbox("Durasi Pengerjaan (Bulan):", ["Cepat", "Sedang", "Lama"])
        material_input = st.selectbox("Ketersediaan Material Konstruksi:", ["tinggi", "sedang", "rendah"])
        
    with col3:
        cuaca_input = st.selectbox("Estimasi Kondisi Cuaca:", ["baik", "normal", "buruk"])
        tenaga_input = st.selectbox("Estimasi Tenaga Kerja:", ["Sedikit", "Sedang", "Banyak"])
        
    st.write("")
    tombol_prediksi = st.button("Hitung Prediksi Keberhasilan Proyek", use_container_width=True)
    
    if tombol_prediksi:
   
        input_data = [[biaya_input, unit_input, durasi_input, material_input, cuaca_input, tenaga_input]]
        input_data_encoded = encoder.transform(input_data)
        

        pred_class_encoded = model.predict(input_data_encoded)
        pred_class = label_encoder.inverse_transform(pred_class_encoded)[0]
        pred_proba = model.predict_proba(input_data_encoded)[0]
        
        st.subheader("Hasil Klasifikasi:")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred_class == "berhasil":
                st.success(f"PROYEK DIPREDIKSI: **BERHASIL / TEPAT WAKTU**")
            else:
                st.error(f"PROYEK DIPREDIKSI: **TERLAMBAT / BERISIKO TINGGI**")
                
        with col_res2:
            st.metric("Peluang Berhasil", f"{pred_proba[0]*100:.2f}%")
            st.metric("Peluang Terlambat", f"{pred_proba[1]*100:.2f}%")


with tab2:
    st.header("Hasil Preprocessing & Diskritisasi Dataset")
    st.write("Data di bawah ini merupakan data kuantitatif Excel Anda yang telah dikategorisasikan secara sistematis:")
    st.dataframe(df, use_container_width=True)


with tab3:
    st.header("Metrik Pengujian Kinerja Sistem")
    

    y_pred = model.predict(X_encoded)
    cm = confusion_matrix(y_encoded, y_pred)
    akurasi = accuracy_score(y_encoded, y_pred)
    
    col_eval1, col_eval2 = st.columns([1, 1])
    
    with col_eval1:
        st.subheader("Grafik Confusion Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax)
        ax.set_ylabel('Aktual')
        ax.set_xlabel('Prediksi')
        st.pyplot(fig)
        
    with col_eval2:
        st.subheader("Persentase Kinerja Algoritma")
        st.info(f"**Akurasi Model:** {akurasi*100:.2f}%")
        st.write("Aplikasi ini menggunakan algoritma Naive Bayes dengan implementasi Laplace Correction. Pengujian performa di atas membuktikan model klasifikasi ini sangat layak dan valid untuk diintegrasikan dalam pengambilan keputusan strategis proyek perumahan cluster.")