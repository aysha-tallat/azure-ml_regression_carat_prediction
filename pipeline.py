from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent
from azure.identity import DefaultAzureCredential


ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="Subscription-ID",
    resource_group_name="rg-dp100",
    workspace_name="mlws-dp100",
)

preprocess_component = CommandComponent(
    name="diamond_preprocess",
    display_name="Diamond Preprocess",
    inputs={
        "train_data": Input(type="uri_folder"),
        "test_data": Input(type="uri_folder"),
    },
    outputs={
        "train_output": Output(type="uri_folder"),
        "test_output": Output(type="uri_folder"),
    },
    code="./",
    command=(
        "python preprocess.py "
        "--train_data ${{inputs.train_data}} "
        "--test_data ${{inputs.test_data}} "
        "--train_output ${{outputs.train_output}} "
        "--test_output ${{outputs.test_output}}"
    ),
    environment="diamonds-env:1",
    compute="Aml-Compute",
)

train_component = CommandComponent(
    name="diamond_train",
    display_name="Diamond Train RF",
    inputs={
        "train_data": Input(type="uri_folder"),
        "test_data": Input(type="uri_folder"),
    },
    outputs={
        "model_output": Output(type="uri_folder"),
    },
    code="./",
    command=(
        "python train.py "
        "--train_data ${{inputs.train_data}} "
        "--test_data ${{inputs.test_data}} "
        "--model_output ${{outputs.model_output}}"
    ),
    environment="diamonds-env:1",
    compute="Aml-Compute",
)


@pipeline(default_compute="Aml-Compute")
def diamond_rf_pipeline():
    preprocess_step = preprocess_component(
        train_data=Input(type="mltable", path="diamond_train"),
        test_data=Input(type="mltable", path="diamond_test"),
    )

    train_step = train_component(
        train_data=preprocess_step.outputs.train_output,
        test_data=preprocess_step.outputs.test_output,
    )

    return {
        "model_output": train_step.outputs.model_output
    }


pipeline_job = diamond_rf_pipeline()
pipeline_job.experiment_name = "diamond-rf-carat"

ml_client.jobs.create_or_update(pipeline_job)
