import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
import os


# ============================================================
# 1. KONFIGURASI FITUR (DAFTAR INPUT)
# ============================================================

# Fitur numerik float
DENSE_FLOAT_FEATURE_KEYS = ['BMI', 'DiabetesPedigreeFunction']

# Fitur numerik integer
DENSE_INT_FEATURE_KEYS = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'Age'
]

# Label untuk klasifikasi
LABEL_KEY = 'Outcome'

# Konfigurasi training
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
TRAIN_STEPS = 100
EPOCHS = 5


# Helper untuk nama fitur hasil transformasi
def _transformed_name(key):
    return key + '_xf'


# ============================================================
# 2. FUNGSI PREPROCESSING (dipakai oleh TFX Transform)
#    TFX akan menjalankan fungsi ini pada data mentah
# ============================================================
def preprocessing_fn(inputs):
    """
    Mengambil dictionary input, lalu melakukan scaling dan bucketize.
    Output kembali ke dictionary baru berisi fitur-fitur hasil transformasi.
    """
    
    outputs = {}

    # Normalisasi fitur numerik dengan Z-score (mean=0, std=1)
    for key in DENSE_FLOAT_FEATURE_KEYS + DENSE_INT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # Membuat fitur kategori berdasarkan umur (5 bucket)
    outputs['Age_bucket'] = tft.bucketize(inputs['Age'], num_buckets=5)

    # Pastikan label bertipe int64
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs


# ============================================================
# 3. DATA LOADER (konversi TFRecord → input model)
# ============================================================
def _input_fn(file_pattern, tf_transform_output, batch_size):
    """
    Memuat TFRecord hasil transform, mem-parse,
    mengembalikan dataset untuk training / evaluasi.
    """

    # Expand file pattern menjadi list file
    file_list = []
    for fp in file_pattern:
        file_list.extend(tf.io.gfile.glob(fp))

    # Load spesifikasi fitur yang sudah ditransformasi
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    # Membaca TFRecord terkompresi GZIP
    dataset = tf.data.TFRecordDataset(file_list, compression_type='GZIP')

    # Parsing tiap record
    def parse_function(proto):
        parsed_features = tf.io.parse_single_example(proto, transformed_feature_spec)
        label = parsed_features.pop(LABEL_KEY)  # ambil label
        return parsed_features, label

    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset


# ============================================================
# 4. MEMBANGUN MODEL KERAS (fix untuk Keras 3 + TFX)
# ============================================================
def _build_keras_model(tf_transform_output):
    """
    Membangun arsitektur DNN untuk klasifikasi biner.
    Menangani fitur numerik + categorical bucketized.
    """

    feature_spec = tf_transform_output.transformed_feature_spec()

    inputs = {}
    processed_inputs = []

    for key, spec in feature_spec.items():

        # Skip label (jangan dijadikan input)
        if key == LABEL_KEY:
            continue

        # Tangani fitur bucket (Age_bucket)
        if key == 'Age_bucket':
            # Input integer 0–4
            inp = tf.keras.Input(shape=(1,), name=key, dtype=tf.int64)
            
            # One-hot encoding (5 kategori)
            x = tf.keras.layers.CategoryEncoding(
                num_tokens=5,
                output_mode="one_hot"
            )(inp)

            # Bentuk output menjadi (5,)
            x = tf.keras.layers.Reshape((5,))(x)

            inputs[key] = inp
            processed_inputs.append(x)

        else:
            # Input untuk fitur numerik float
            inp = tf.keras.Input(shape=(1,), name=key, dtype=tf.float32)

            # Casting untuk menghindari error dtype di Keras 3
            x = tf.keras.layers.Lambda(
                lambda t: tf.cast(t, tf.float32)
            )(inp)

            inputs[key] = inp
            processed_inputs.append(x)

    if len(processed_inputs) == 0:
        raise ValueError("Tidak ada fitur yang ditemukan dari Transform!")

    # Gabungkan semua fitur menjadi satu tensor
    concat = tf.keras.layers.Concatenate()(processed_inputs)

    # Hidden layers
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Output layer (sigmoid → binary classification)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model


# ============================================================
# 5. FUNGSI UTAMA UNTUK TFX TRAINER (run_fn)
# ============================================================
def run_fn(fn_args):
    """
    Fungsi ini dijalankan oleh komponen Trainer pada pipeline TFX.
    Tugasnya:
    - Load transform_output
    - Bangun model Keras
    - Load train & eval dataset
    - Training
    - Export model ke folder serving_model_dir
    """

    print("=== Mulai Training Model ===")

    # Load hasil transform
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Siapkan dataset training & evaluasi
    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        TRAIN_BATCH_SIZE
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        EVAL_BATCH_SIZE
    )

    # Bangun model
    model = _build_keras_model(tf_transform_output)

    # Training model
    model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS
    )

    print("=== Training Selesai. Mengekspor Model ===")

    # Export untuk serving (TFX Pusher)
    model.export(fn_args.serving_model_dir)
