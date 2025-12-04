import tensorflow_model_analysis as tfma
import os
import pprint

# Lokasi output Evaluator
output_path = 'output/Evaluator/evaluation'

if not os.path.exists(output_path):
    print("Error: Folder evaluasi tidak ditemukan! Jalankan pipeline dulu.")
    exit()

# Cari folder hasil run paling baru
runs = [os.path.join(output_path, d) for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
if not runs:
    print("Belum ada hasil evaluasi.")
    exit()

latest_run = max(runs, key=os.path.getmtime)
print(f"=== Membaca Hasil Evaluasi dari: {latest_run} ===")

# Load hasil
try:
    eval_result = tfma.load_eval_result(latest_run)
    slicing_metrics = eval_result.slicing_metrics
    
    print("\n--- NILAI RAPOR MODEL ---")
    
    for item in slicing_metrics:
        # DETEKSI FORMAT: Apakah Tuple atau Objek?
        if isinstance(item, tuple):
            # Jika Tuple: (slice_key, metrics)
            slice_key, metrics = item
        else:
            # Jika Objek
            slice_key = item.slice_key
            metrics = item.metrics
            
        # Kita hanya cari Slice Kosong () yang artinya OVERALL (Rata-rata Keseluruhan)
        # Slice key biasanya berbentuk tuple list, misal: (('Age_bucket', 1),)
        # Jika kosong (), berarti itu Overall.
        
        is_overall = False
        if len(slice_key) == 0:
            is_overall = True
            
        if is_overall:
            print(f"Slice: OVERALL (Keseluruhan Data)")
            
            # Print Binary Accuracy & AUC
            # Kita loop dictionary metrics untuk mencari kuncinya
            found_metrics = False
            for k, v in metrics.items():
                # Ubah key jadi string biar aman (kadang tuple)
                key_str = str(k).lower()
                
                # Filter hanya metric penting
                if 'binary_accuracy' in key_str or 'auc' in key_str:
                    # Ambil valuenya. Kadang terbungkus dalam dict {'value': 0.7}
                    val = v
                    if isinstance(v, dict) and 'value' in v:
                        val = v['value']
                    
                    # Print hasil
                    print(f" -> {k}: {val}")
                    found_metrics = True
                    
                    # Cek Lulus/Gagal khusus Binary Accuracy
                    if 'binary_accuracy' in key_str:
                        try:
                            score = float(val)
                            if score < 0.6:
                                print(f"    ❌ GAGAL: {score:.4f} < 0.6 (Threshold)")
                            else:
                                print(f"    ✅ LULUS: {score:.4f} > 0.6 (Threshold)")
                        except:
                            pass

            if not found_metrics:
                print("   [!] Tidak menemukan metric binary_accuracy/auc. Ini daftar raw metrics:")
                pprint.pprint(metrics)
                
            break # Sudah ketemu Overall, stop looping

except Exception as e:
    print(f"Gagal membaca metrics (Error Detail): {e}")
    import traceback
    traceback.print_exc()