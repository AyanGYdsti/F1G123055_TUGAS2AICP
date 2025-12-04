# pipeline.py
import os
import logging
import tfx
from tfx.proto import trainer_pb2, example_gen_pb2, pusher_pb2
import tensorflow_model_analysis as tfma
from tfx.orchestration.local import local_dag_runner
from tfx.orchestration import metadata
from tfx.orchestration.pipeline import Pipeline

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)

logging.getLogger().setLevel(logging.INFO)

PIPELINE_NAME = 'diabetes_pipeline_complete'
PIPELINE_ROOT = 'output'
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata.db')

# **Pisahkan file module**: transform dan trainer sebaiknya berbeda
TRANSFORM_MODULE_FILE = os.path.join('modules', 'diabetes_transform.py')
TRAINER_MODULE_FILE = os.path.join('modules', 'diabetes_trainer.py')

DATA_ROOT = 'data'  # pastikan folder ini hanya berisi dataset yang akan dibaca
SERVING_MODEL_DIR = os.path.join('serving_model')

def create_pipeline():

    # 1) ExampleGen
    example_gen = CsvExampleGen(
        input_base=DATA_ROOT,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            ])
        )
    )

    # 2) Statistics
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # 3) Schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )

    # 4) ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5) Transform  (pakai module transform khusus)
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=TRANSFORM_MODULE_FILE
    )

    # 6) Trainer
    # NOTE: Trainer expects 'examples' (channel). Kita beri channel hasil Transform.
    trainer = Trainer(
        module_file=TRAINER_MODULE_FILE,
        examples=transform.outputs['transformed_examples'],   # <- BENAR
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    # 7) Evaluator (pakai transformed_examples supaya cocok dengan model input)
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(signature_name='serving_default', label_key='Outcome')
        ],
        slicing_specs=[
            tfma.SlicingSpec(),  # overall
            tfma.SlicingSpec(feature_keys=['Age_bucket'])  # slice by Age bucket
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy',
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
        examples=transform.outputs['transformed_examples'],  # penting: transformed
        model=trainer.outputs['model'],
        eval_config=eval_config
    )

    # 8) Pusher: hanya push jika model blessed
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        )
    )

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

    return Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
        enable_cache=True
    )


if __name__ == '__main__':
    # optional: bersihkan hasil run sebelumnya
    if os.path.exists('output'):
        import shutil
        shutil.rmtree('output', ignore_errors=True)
    if os.path.exists('serving_model'):
        import shutil
        shutil.rmtree('serving_model', ignore_errors=True)

    local_dag_runner.LocalDagRunner().run(create_pipeline())
    print("Selesai.")
