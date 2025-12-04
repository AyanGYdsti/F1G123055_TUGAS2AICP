import tensorflow as tf
import numpy as np
import os

# 1. Cari lokasi model hasil Pusher
# Kita ambil folder timestamp paling baru di dalam serving_model
serving_dir = 'serving_model'
if not os.path.exists(serving_dir):
    raise Exception("Folder serving_model tidak ditemukan! Jalankan pipeline dulu.")

timestamps = os.listdir(serving_dir)
if not timestamps:
    raise Exception("Model belum dipush oleh Pusher!")

latest_model_path = os.path.join(serving_dir, max(timestamps))
print(f"=== Memuat Model dari: {latest_model_path} ===")

# 2. Load Model (Keras 3 SavedModel)
model = tf.saved_model.load(latest_model_path)

# 3. Siapkan Data Dummy (Format Transformed / Z-Score)
# Ingat: Model ini mengharapkan input matang karena kita export langsung dari Keras
# Kita buat data seolah-olah sudah dinormalisasi (Range -1 s.d 1)

# Fitur Float (Z-Score)
inference_data = {
    'BMI_xf': tf.constant([[0.5]], dtype=tf.float32),                 # Agak gemuk
    'DiabetesPedigreeFunction_xf': tf.constant([[-0.2]], dtype=tf.float32),
    'Pregnancies_xf': tf.constant([[1.0]], dtype=tf.float32),         # Sering hamil
    'Glucose_xf': tf.constant([[1.5]], dtype=tf.float32),             # Gula tinggi (bahaya)
    'BloodPressure_xf': tf.constant([[0.1]], dtype=tf.float32),
    'SkinThickness_xf': tf.constant([[-0.5]], dtype=tf.float32),
    'Insulin_xf': tf.constant([[0.8]], dtype=tf.float32),
    'Age_xf': tf.constant([[1.2]], dtype=tf.float32),
    
    # Fitur Bucket (Age_bucket) - Harus Integer
    # Ingat logic kita: CategoryEncoding one-hot
    'Age_bucket': tf.constant([[3]], dtype=tf.int64) # Kelompok umur tua
}

print("\n=== Melakukan Prediksi ===")
try:
    # Keras 3 export biasanya menghasilkan fungsi 'serve' atau callable langsung
    # Kita coba panggil default serving signature
    inference_func = model.signatures['serving_default']
    
    # Lakukan prediksi
    predictions = inference_func(**inference_data)
    
    # Ambil hasil (biasanya key-nya 'output_0' atau 'dense_x')
    # Kita print semua keys outputnya
    print("Output Keys:", list(predictions.keys()))
    
    # Ambil nilai probabilitas
    # Karena output layer cuma 1 node (Sigmoid), ambil key pertama
    output_key = list(predictions.keys())[0]
    probabilitas = predictions[output_key].numpy()[0][0]
    
    print(f"\nProbabilitas Diabetes: {probabilitas:.4f} ({probabilitas*100:.2f}%)")
    
    if probabilitas > 0.5:
        print("Kesimpulan: POSITIF Diabetes ⚠️")
    else:
        print("Kesimpulan: NEGATIF Diabetes ✅")
        
except Exception as e:
    print(f"Error saat inferensi: {e}")
    print("Coba cek input shape model Anda.")