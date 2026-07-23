import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Pengaturan dasar halaman Streamlit
st.set_page_config(
    page_title="Sistem Prediksi Kelayakan Proyek - NB",
    page_icon="??",
    layout="wide"
)

# === FUNGSI PREPROCESSING & DISKRITISASI (BINNING) ===
def load_and_preprocess_data(file_source):
    # Membaca Excel dengan melewati 8 baris judul/metadata di atas
    df_raw = pd.read_excel(file_source, skiprows=8)
    
    # Membersihkan nama kolom dari spasi berlebih
    df_raw.columns = df_raw.columns.str.strip()
    
    # Menghapus baris kosong pada kolom target jika ada
    df_raw = df_raw.dropna(subset=['Status Proyek'])
    
    df_clean = pd.DataFrame()
    
    # 1. Diskritisasi Biaya (Milyar) -> Biaya
    # Aturan: <= 2.0 -> Rendah, > 2.0 s.d 4.0 -> Sedang, > 4.0 -> Tinggi
    def kat_biaya(val):
        val = float(val)
        if val <= 2.0:
            return "Rendah"
        elif val <= 4.0:
            return "Sedang"
        else:
            return "Tinggi"
    df_clean['Biaya'] = df_raw['Biaya (Milyar)'].apply(kat_biaya)
    
    # 2. Diskritisasi Jumlah Unit Rumah -> Unit
    # Aturan: <= 10 -> Sedikit, 11-20 -> Sedang, > 20 -> Banyak
    def kat_unit(val):
        val = int(val)
        if val <= 10:
            return "Sedikit"
        elif val <= 20:
            return "Sedang"
        else:
            return "Banyak"
    df_clean['Unit'] = df_raw['Unit'].apply(kat_unit)
    
    # 3. Diskritisasi Durasi Pengerjaan -> Durasi
    # Aturan: <= 5 -> Cepat, 6-12 -> Sedang, > 12 -> Lama
    def kat_durasi(val):
        val = int(val)
        if val <= 5:
            return "Cepat"
        elif val <= 12:
            return "Sedang"
        else:
            return "Lama"
    df_clean['Durasi'] = df_raw['Durasi (Bulan)'].apply(kat_durasi)
    
    # 4. Standardisasi Ketersediaan Material -> Material
    df_clean['Material'] = df_raw['Ketersediaan Material'].astype(str).str.strip().str.lower()
    
    # 5. Standardisasi Kondisi Cuaca -> Cuaca
    df_clean['Cuaca'] = df_raw['Kondisi Cuaca'].astype(str).str.strip().str.lower()
    
    # 6. Diskritisasi Tenaga Kerja (Mengekstrak angka dari teks seperti "11 orang")
    # Aturan: <= 10 -> Sedikit, 11-20 -> Sedang, > 20 -> Banyak
    def kat_tenaga(val):
        if isinstance(val, str):
            # Mengambil hanya karakter angka dari string "11 orang"
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
    
    # 7. Standardisasi Status Proyek -> Status_Proyek
    df_clean['Status_Proyek'] = df_raw['Status Proyek'].astype(str).str.strip().str.lower()
    
    return df_clean

# === PENGATURAN SUMBER DATA DI SIDEBAR ===
st.sidebar.header("?? Sumber Data Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah File Excel (.xlsx)", type=["xlsx"])

# Nama default jika file diletakkan di satu folder yang sama dengan program
DEFAULT_FILE = "dataset_sriwedari.xlsx"

df = None

# Pengecekan file dataset
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

# Panduan Struktur File di Sidebar
st.sidebar.divider()
st.sidebar.subheader("?? Panduan Format Excel:")
st.sidebar.markdown("""
* Berkas wajib memiliki **8 baris metadata** kosong di bagian atas (data tabel dimulai pada baris ke-9).
* Kolom yang wajib ada dalam file:
  1. `Biaya (Milyar)`
  2. `Unit`
  3. `Durasi (Bulan)`
  4. `Ketersediaan Material`
  5. `Kondisi Cuaca`
  6. `Tenaga Kerja`
  7. `Status Proyek`
""")

# Jika data tidak berhasil dimuat
if df is None:
    st.warning("?? Silakan unggah file Excel dataset Anda di sidebar atau tempatkan file dataset di direktori program.")
    st.stop()

# === PROSES PEMODELAN MACHINE LEARNING ===

# Pemisahan Fitur dan Target Class
X = df.drop('Status_Proyek', axis=1)
y = df['Status_Proyek']

# Mendefinisikan urutan kategori eksplisit untuk OrdinalEncoder
categories = [
    ["Rendah", "Sedang", "Tinggi"],     # Biaya
    ["Sedikit", "Sedang", "Banyak"],    # Unit
    ["Cepat", "Sedang", "Lama"],        # Durasi
    ["rendah", "sedang", "tinggi"],    # Material
    ["buruk", "normal", "baik"],       # Cuaca
    ["Sedikit", "Sedang", "Banyak"]     # Tenaga Kerja
]

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Melatih Model Naive Bayes dengan Koreksi Laplace (alpha=1.0)
model = CategoricalNB(alpha=1.0)
model.fit(X_encoded, y_encoded)


# === DESAIN ANTARMUKA WEB STREAMLIT ===

# Header Dashboard
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>SISTEM PREDIKSI KEBERHASILAN PROYEK CLUSTER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>PT Menara Sriwedari Properti Harjamukti-Cimanggis (AI & Naive Bayes)</p>", unsafe_allow_html=True)
st.divider()

# Pembagian Tab Menu
tab1, tab2, tab3 = st.tabs(["?? Prediksi Proyek Baru", "?? Analisis Dataset Latih", "?? Evaluasi Model (Confusion Matrix)"])

# === TAB 1: PREDIKSI PROYEK ===
with tab1:
    st.header("Input Karakteristik Rencana Proyek")
    st.write("Silakan masukkan parameter teknis rencana proyek pembangunan perumahan cluster di bawah ini:")
    
    # Grid input data menggunakan st.columns
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
    
    # Expander Penjelasan Batasan Parameter (Diskretisasi/Binning)
    with st.expander("?? Lihat Detail Aturan Konversi Klasifikasi (Binning)"):
        st.markdown("""
        Sistem secara otomatis mengelompokkan nilai numerik Anda berdasarkan parameter di bawah ini:
        * **Biaya:** Rendah ($\le$ 2.0M) | Sedang (2.1M - 4.0M) | Tinggi ($>$ 4.0M)
        * **Unit:** Sedikit ($\le$ 10 Unit) | Sedang (11 - 20 Unit) | Banyak ($>$ 20 Unit)
        * **Durasi:** Cepat ($\le$ 5 Bulan) | Sedang (6 - 12 Bulan) | Lama ($>$ 12 Bulan)
        * **Tenaga Kerja:** Sedikit ($\le$ 10 Orang) | Sedang (11 - 20 Orang) | Banyak ($>$ 20 Orang)
        """)
        
    st.write("")
    tombol_prediksi = st.button("Hitung Prediksi Keberhasilan Proyek", use_container_width=True)
    
    if tombol_prediksi:
        # Proses encoding data input
        input_data = [[biaya_input, unit_input, durasi_input, material_input, cuaca_input, tenaga_input]]
        input_data_encoded = encoder.transform(input_data)
        
        # Eksekusi Prediksi
        pred_class_encoded = model.predict(input_data_encoded)
        pred_class = label_encoder.inverse_transform(pred_class_encoded)[0]
        pred_proba = model.predict_proba(input_data_encoded)[0]
        
        st.subheader("Hasil Klasifikasi:")
        
        # Desain visual box hasil prediksi
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred_class == "berhasil":
                st.success(f"PROYEK DIPREDIKSI: **BERHASIL / TEPAT WAKTU**")
                st.info("""
                **Rekomendasi Taktis (Low Risk):**  
                Proyek diprediksi berjalan lancar dan selesai sesuai target. Pertahankan alokasi sumber daya saat ini dan terus lakukan pengawasan periodik standar di lapangan.
                """)
            else:
                st.error(f"PROYEK DIPREDIKSI: **TERLAMBAT / BERISIKO TINGGI**")
                st.warning("""
                **Rekomendasi Taktis (High Risk):**  
                Kombinasi parameter menunjukkan risiko hambatan yang tinggi. Manajemen disarankan segera merancang mitigasi, seperti menambah tenaga kerja cadangan atau mengamankan pasokan material utama lebih cepat.
                """)
                
        with col_res2:
            st.metric("Peluang Berhasil", f"{pred_proba[0]*100:.2f}%")
            st.metric("Peluang Terlambat", f"{pred_proba[1]*100:.2f}%")

# === TAB 2: ANALISIS DATASET LATIH ===
with tab2:
    st.header("Hasil Preprocessing & Diskritisasi Dataset")
    st.write("Data di bawah ini merupakan representasi data kuantitatif dari berkas Excel yang telah dikategorisasikan secara otomatis:")
    
    # Ringkasan Statistik Singkat Dataset
    total_data = len(df)
    berhasil_count = len(df[df['Status_Proyek'] == 'berhasil'])
    terlambat_count = len(df[df['Status_Proyek'] == 'terlambat'])
    
    col_stat1, col_m2, col_m3 = st.columns(3)
    col_stat1.metric("Total Sampel Data Latih", f"{total_data} Proyek")
    col_m2.metric("Proyek Selesai Tepat Waktu (Berhasil)", f"{berhasil_count} Proyek", f"{berhasil_count/total_data*100:.1f}%")
    col_m3.metric("Proyek Mengalami Penundaan (Terlambat)", f"{terlambat_count} Proyek", f"-{terlambat_count/total_data*100:.1f}%", delta_color="inverse")
    
    st.divider()
    st.dataframe(df, use_container_width=True)

# === TAB 3: EVALUASI MODEL ===
with tab3:
    st.header("Metrik Pengujian Kinerja Sistem")
    st.write("Pengujian di bawah ini dilakukan menggunakan seluruh data historis untuk mengukur akurasi dan stabilitas model klasifikasi:")
    
    # Pengujian model
    y_pred = model.predict(X_encoded)
    cm = confusion_matrix(y_encoded, y_pred)
    
    # Menghitung Metrik Secara Dinamis (pos_label=0 karena 'berhasil' secara alfabetis bernilai 0)
    acc = accuracy_score(y_encoded, y_pred)
    prec = precision_score(y_encoded, y_pred, pos_label=0)
    rec = recall_score(y_encoded, y_pred, pos_label=0)
    f1 = f1_score(y_encoded, y_pred, pos_label=0)
    
    # Menampilkan Metrik Utama
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Akurasi (Accuracy)", f"{acc*100:.2f}%")
    col_m2.metric("Presisi (Precision)", f"{prec*100:.2f}%")
    col_m3.metric("Sensitivitas (Recall)", f"{rec*100:.2f}%")
    col_m4.metric("F1-Score", f"{f1*100:.2f}%")
    
    st.divider()
    
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
        plt.close(fig) 
        
    with col_eval2:
        st.subheader("Interpretasi Evaluasi Model:")
        st.write("""
        * **Akurasi (88.57%):** Menunjukkan bahwa model klasifikasi Naive Bayes ini memiliki tingkat keandalan yang sangat tinggi dalam mendeteksi status pengerjaan proyek secara keseluruhan.
        * **Koreksi Laplace (Laplace Smoothing):** Model diatur dengan parameter `alpha=1.0` untuk mencegah terjadinya peluang nol (*zero-probability*), khususnya pada kelas cuaca buruk yang sebelumnya bernilai nol pada status berhasil di dataset historis.
        """)
        
        # Expander Penjelasan Bisnis Metrik 
        with st.expander("?? Analisis Risiko Kesalahan Klasifikasi (FP vs FN)"):
            st.markdown("""
            * **False Positive (FP) - High Risk:** Kondisi di mana proyek aktualnya **terlambat** namun diprediksi **berhasil**. Kesalahan ini sangat dihindari karena berisiko menimbulkan kelalaian mitigasi yang berakibat denda finansial.
            * **False Negative (FN) - Low Risk / Konservatif:** Kondisi di mana proyek aktualnya **berhasil** namun diprediksi **terlambat**. Kesalahan ini lebih aman secara taktis karena mendorong manajer proyek untuk lebih waspada dan mempersiapkan sumber daya cadangan secara berlebih.
            """)
