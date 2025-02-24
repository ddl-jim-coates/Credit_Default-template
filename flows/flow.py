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
# pyflyte run --remote ./flows/flow.py model_training_flow --data_path /mnt/data/workshop_dev/credit_card_default.csv


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
        output_specs=[Output(name="model", type=sklearn_log_regArtifact.File(name="model.pkl")),
                      Output(name="log_reg_ROC_Curve_C", type=FlyteFile[TypeVar('png')]),
                      Output(name="log_reg_confusion_matrix_C", type=FlyteFile[TypeVar('png')]),
                      Output(name="log_reg_precision_recall_C", type=FlyteFile[TypeVar('png')]),
                      Output(name="log_reg_ROC_Curve", type=FlyteFile[TypeVar('png')]),
                      Output(name="log_reg_confusion_matrix", type=FlyteFile[TypeVar('png')]),
                      Output(name="log_reg_precision_recall", type=FlyteFile[TypeVar('png')]),
                      Output(name="log_reg_confusion_matrix", type=FlyteFile[TypeVar('png')])],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name
    )

    # Launch H2O model training
    h2o_results = run_domino_job_task(
        flyte_task_name="Train H2O Model",
        command="flows/h2o_model_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default'])],
        output_specs=[Output(name="model", type=h2oArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
    )

    # Launch sklearn random forest training
    sklearn_rf_results = run_domino_job_task(
        flyte_task_name="Train Sklearn RF",
        command="flows/sklearn_RF_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default'])],
        output_specs=[Output(name="model", type=sklearn_rfArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
    )

    # Launch XGBoost model training
    xgboost_results = run_domino_job_task(
        flyte_task_name="Train XGBoost",
        command="flows/xgb_model_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default'])],
        output_specs=[Output(name="model", type=xgboostArtifact.File(name="model.pkl"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
    )


    return