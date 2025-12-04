import tensorflow as tf
import tensorflow_transform as tft

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

    for key in DENSE_FLOAT_FEATURE_KEYS + DENSE_INT_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    outputs['Age_bucket'] = tft.bucketize(inputs['Age'], num_buckets=5)

    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs