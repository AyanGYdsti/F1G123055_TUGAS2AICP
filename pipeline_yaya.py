# pipeline.py
# Pipeline TFX lengkap untuk dataset Diabetes:
# - Meliputi ingestion, statistik, schema inference, validasi, transformasi,
#   training model, evaluasi, dan push model ke direktori serving.

import os
import logging
import tfx
from tfx.proto import trainer_pb2, example_gen_pb2, pusher_pb2
import tensorflow_model_analysis as tfma
from tfx.orchestration.local import local_dag_runner
from tfx.orchestration import metadata
from tfx.orchestration.pipeline import Pipeline

from tfx.components import (
    CsvExampleGen,       # baca data CSV menjadi TFRecord
    StatisticsGen,       # generate statistik dataset
    SchemaGen,           # infer schema otomatis dari statistik
    ExampleValidator,    # validasi anomali data terhadap schema
    Transform,           # feature engineering menggunakan tf.Transform
    Trainer,             # training model TensorFlow
    Evaluator,           # evaluasi model & model blessing via TFMA
    Pusher,              # push model hanya jika lolos evaluasi
)

logging.getLogger().setLevel(logging.INFO)

PIPELINE_NAME = 'diabetes_pipeline_complete'
PIPELINE_ROOT = 'output'
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata.db')

# File transform dan trainer dipisahkan agar modular
TRANSFORM_MODULE_FILE = os.path.join('modules', 'diabetes_transform.py')
TRAINER_MODULE_FILE = os.path.join('modules', 'diabetes_trainer.py')

DATA_ROOT = 'data'  # direktori dataset CSV
SERVING_MODEL_DIR = os.path.join('serving_model')

def create_pipeline():

    # 1) ExampleGen: baca CSV dan lakukan pembagian data otomatis berbasis hash
    example_gen = CsvExampleGen(
        input_base=DATA_ROOT,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            ])
        )
    )

    # 2) StatisticsGen: menghasilkan statistik deskriptif untuk data
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # 3) SchemaGen: membuat schema otomatis berdasarkan statistik
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )

    # 4) ExampleValidator: cek anomali data (missing, out-of-range, dll)
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5) Transform: feature engineering menggunakan tf.Transform
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=TRANSFORM_MODULE_FILE
    )

    # 6) Trainer: training model TensorFlow menggunakan data yang sudah ditransform
    trainer = Trainer(
        module_file=TRAINER_MODULE_FILE,
        examples=transform.outputs['transformed_examples'],  # wajib: pakai transformed examples
        transform_graph=transform.outputs['transform_graph'], # graph transform untuk preprocessing di serving
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    # 7) Evaluator: evaluasi model menggunakan TFMA, dengan slicing dan threshold
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(signature_name='serving_default', label_key='Outcome')
        ],
        slicing_specs=[
            tfma.SlicingSpec(),                      # overall metric
            tfma.SlicingSpec(feature_keys=['Age_bucket'])  # slice berdasarkan bucket umur
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                # metric BinaryAccuracy dengan threshold minimal 0.5
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}
                        )
                    )
                ),
                tfma.MetricConfig(class_name='AUC')
            ])
        ]
    )

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config
    )

    # 8) Pusher: push model hanya jika evaluator memberikan "blessing"
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],  # hanya push jika lolos threshold
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        )
    )

    # Daftar komponen yang dijalankan pipeline
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        evaluator,
        pusher
    ]

    # Instansiasi pipeline TFX
    return Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
        enable_cache=True   # menghindari re-run komponen yang sudah ada
    )


if __name__ == '__main__':
    # Optional: bersihkan output run sebelumnya (agar pipeline fresh)
    if os.path.exists('output'):
        import shutil
        shutil.rmtree('output', ignore_errors=True)

    if os.path.exists('serving_model'):
        import shutil
        shutil.rmtree('serving_model', ignore_errors=True)

    print("Menjalankan pipeline TFX end-to-end...")
    local_dag_runner.LocalDagRunner().run(create_pipeline())
    print("Pipeline Selesai.")
