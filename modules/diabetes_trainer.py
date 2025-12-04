import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
import os

# --- 1. KONFIGURASI FITUR ---
DENSE_FLOAT_FEATURE_KEYS = ['BMI', 'DiabetesPedigreeFunction']
DENSE_INT_FEATURE_KEYS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age']
LABEL_KEY = 'Outcome'

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
TRAIN_STEPS = 100
EPOCHS = 5

def _transformed_name(key):
    return key + '_xf'

# --- 2. FUNGSI PREPROCESSING ---
def preprocessing_fn(inputs):
    outputs = {}
    for key in DENSE_FLOAT_FEATURE_KEYS + DENSE_INT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
    
    outputs['Age_bucket'] = tft.bucketize(inputs['Age'], num_buckets=5)
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs

# --- 3. FUNGSI DATA LOADER ---
def _input_fn(file_pattern, tf_transform_output, batch_size):
    # file_pattern = ['path1', 'path2', ...]
    file_list = []
    for fp in file_pattern:
        file_list.extend(tf.io.gfile.glob(fp))

    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    
    dataset = tf.data.TFRecordDataset(file_list, compression_type='GZIP')

    def parse_function(proto):
        parsed_features = tf.io.parse_single_example(proto, transformed_feature_spec)
        label = parsed_features.pop(LABEL_KEY)
        return parsed_features, label

    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset

# --- 4. ARSITEKTUR MODEL (PERBAIKAN UTAMA DI SINI) ---
def _build_keras_model(tf_transform_output):
    feature_spec = tf_transform_output.transformed_feature_spec()

    inputs = {}
    processed_inputs = []

    for key, spec in feature_spec.items():
        # HANYA skip Label. Age_bucket JANGAN di-skip.
        if key == LABEL_KEY:
            continue

        if key == 'Age_bucket':
            # Age_bucket output dari Transform biasanya Int64
            # Kita buat input layer khusus integer
            inp = tf.keras.Input(shape=(1,), name=key, dtype=tf.int64)
            
            # Lakukan One-Hot Encoding (5 kategori)
            x = tf.keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")(inp)
            x = tf.keras.layers.Reshape((5,))(x) # Ratakan hasil one-hot
            
            inputs[key] = inp
            processed_inputs.append(x)
            
        else:
            # Untuk fitur numeric biasa (Float)
            inp = tf.keras.Input(shape=(1,), name=key, dtype=tf.float32)
            
            # Safe casting untuk Keras 3
            x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(inp)
            
            inputs[key] = inp
            processed_inputs.append(x)

    if len(processed_inputs) == 0:
        raise ValueError("No features found from transform output!")

    concat = tf.keras.layers.Concatenate()(processed_inputs)

    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model


# --- 5. FUNGSI UTAMA (RUN_FN) ---
def run_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

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

    model = _build_keras_model(tf_transform_output)

    model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS
    )

    model.export(fn_args.serving_model_dir)