import streamlit as st
import pandas as pd
import joblib
import os

# Judul aplikasi
st.title('Aplikasi Prediksi Tingkat Stres')
st.write('Aplikasi ini memprediksi tingkat stres berdasarkan kebiasaan digital dan faktor gaya hidup.')

# Memuat model yang telah dilatih
# Pastikan file 'model_stress_predictor.pkl' berada di direktori yang sama dengan 'app.py'
try:
    model = joblib.load('model_stress_predictor.pkl')
    st.success("Model berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: File 'model_stress_predictor.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama dengan aplikasi Streamlit ini.")
    st.stop() # Menghentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop() # Menghentikan eksekusi jika ada error lain saat memuat model

# Input dari pengguna
st.header('Masukkan Data untuk Prediksi')

screen_time = st.slider('Waktu Layar per Hari (jam)', min_value=0.0, max_value=24.0, value=5.0, step=0.5)
tiktok_hours = st.slider('Jam Penggunaan TikTok per Hari (jam)', min_value=0.0, max_value=10.0, value=2.0, step=0.5)
sleep_hours = st.slider('Jam Tidur per Hari (jam)', min_value=0.0, max_value=12.0, value=7.0, step=0.5)
num_platforms = st.slider('Jumlah Platform Media Sosial yang Digunakan', min_value=0, max_value=10, value=3, step=1)
mood_score = st.slider('Skor Suasana Hati (1-10, 10 = sangat baik)', min_value=1, max_value=10, value=7, step=1)

# Tombol untuk memicu prediksi
if st.button('Prediksi Tingkat Stres'):
    # Buat DataFrame dari input pengguna
    new_data = pd.DataFrame(
        [[screen_time, tiktok_hours, sleep_hours, num_platforms, mood_score]],
        columns=['screen_time_hours', 'hours_on_TikTok', 'sleep_hours', 'social_media_platforms_used', 'mood_score']
    )

    # Lakukan prediksi
    predicted_stress = model.predict(new_data)[0]

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    st.success(f'Prediksi Tingkat Stres Anda adalah: **{predicted_stress:.2f}** (skala 0-10)')

    st.write('---')
    st.write('**Catatan:**')
    st.write('- Tingkat stres diprediksi pada skala 0 hingga 10.')
    st.write('- Model ini didasarkan pada data yang digunakan untuk pelatihan dan mungkin tidak 100% akurat untuk setiap individu.')

# Informasi tambahan (opsional)
st.sidebar.header('Tentang Aplikasi Ini')
st.sidebar.info(
    'Aplikasi ini dibangun menggunakan Streamlit dan model Random Forest Regressor yang dilatih '
    'untuk memprediksi tingkat stres berdasarkan kebiasaan digital dan faktor gaya hidup.'
)
