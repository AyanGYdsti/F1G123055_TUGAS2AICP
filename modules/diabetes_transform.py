import tensorflow as tf
import tensorflow_transform as tft

# KUNCI KOMPATIBILITAS TFX 1.16:
# Kita mendefinisikan key fitur secara eksplisit.
DENSE_FLOAT_FEATURE_KEYS = ['BMI', 'DiabetesPedigreeFunction']
DENSE_INT_FEATURE_KEYS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age']
LABEL_KEY = 'Outcome'

def _transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    """
    tf.transform callback function.
    Input: inputs (dict of Tensors)
    Output: outputs (dict of Tensors)
    """
    outputs = {}

    # 1. Normalisasi Z-Score
    # Menggabungkan float dan int karena semuanya butuh dinormalisasi
    for key in DENSE_FLOAT_FEATURE_KEYS + DENSE_INT_FEATURE_KEYS:
        # tft.scale_to_z_score adalah fungsi standar di TFX 1.x
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # 2. Bucketing Age (Umur)
    # Mengubah umur menjadi kategori (0, 1, 2, 3, 4)
    outputs['Age_bucket'] = tft.bucketize(inputs['Age'], num_buckets=5)

    # 3. Casting Label
    # Mengubah label menjadi int64 agar kompatibel dengan Learner nanti
    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs