from flytekit import workflow
from flytekit.types.file import FlyteFile
from typing import TypeVar, NamedTuple
from flytekitplugins.domino.helpers import run_domino_job_task, Input, Output
from flytekitplugins.domino.task import DatasetSnapshot
from flytekitplugins.domino.artifact import Artifact, DATA, MODEL, REPORT
import os

# Set default Compute Environment and Hardware Tier for all tasks. 
environment_name = "Credit Default Environment"
hardware_tier_name = "Small"


# Enter the name of your project's default dataset. 
# Ensure you have taken a snapshot of that dataset in order for it to be mounted to your flow tasks.
dataset_name="workshop_dev"
snapshot_number=1


# Enter the command below to run this Flow. 
# pyflyte run --remote ./flows/flow_dev.py model_training_flow --data_path /mnt/data/workshop_dev/credit_card_default.csv


# Define Flow Artifacts to capture for each model training task
sklearn_log_regArtifact = Artifact("scikit-learn Logistic Regression", MODEL)
h2oArtifact = Artifact("H20 AutoML", MODEL)
sklearn_rfArtifact = Artifact("scikit-learn Random Forest", MODEL)
xgboostArtifact = Artifact("XGBoost", MODEL)


@workflow
def model_training_flow(data_path: str):
    """
    Workflow that runs multiple model training jobs in parallel.
    Returns trained model files for each algorithm as seperate Flow Artifacts.
    """

    # Load csv from Dataset to Flows blob
    load_data = run_domino_job_task(
        flyte_task_name="Load Data",
        command="flows/load_data.py",
        inputs=[Input(name='data_path', type=str, value=data_path)],
        output_specs=[Output(name="credit_card_default", type=FlyteFile[TypeVar('csv')])],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
        dataset_snapshots=[
            DatasetSnapshot(Name=dataset_name, Version=snapshot_number)
        ]
    )

    # Launch sklearn logistic regression training
    sklearn_log_reg_results = run_domino_job_task(
        flyte_task_name="Train Sklearn LogReg",
        command="flows/sklearn_log_reg_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default'])],
        output_specs=[Output(name="model", type=sklearn_log_regArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name
    )


    return