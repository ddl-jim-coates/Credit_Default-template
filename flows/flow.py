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
sklearn_log_regArtifact_charts = Artifact("scikit-learn Logistic Regression Charts", REPORT)

h2oArtifact = Artifact("H20 AutoML", MODEL)
h2oArtifact_charts = Artifact("H20 AutoML Charts", REPORT)

sklearn_rfArtifact = Artifact("scikit-learn Random Forest", MODEL)
sklearn_rfArtifact_charts = Artifact("scikit-learn Random Forest Charts", REPORT)


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
        ],
        cache=True,
        cache_version="1.0"
    )

    # Launch sklearn logistic regression training
    sklearn_log_reg_results = run_domino_job_task(
        flyte_task_name="Train Sklearn LogReg",
        command="flows/sklearn_log_reg_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default'])],
        output_specs=[Output(name="model", type=sklearn_log_regArtifact.File(name="model.pkl")),
                      Output(name="log_reg_ROC_Curve_C", type=sklearn_log_regArtifact_charts.File(name="log_reg_ROC_Curve_C.png")),
                      Output(name="log_reg_confusion_matrix_C", type=sklearn_log_regArtifact_charts.File(name="log_reg_confusion_matrix_C.png")),
                      Output(name="log_reg_precision_recall_C", type=sklearn_log_regArtifact_charts.File(name="log_reg_precision_recall_C.png")),
                      Output(name="log_reg_ROC_Curve", type=sklearn_log_regArtifact_charts.File(name="log_reg_ROC_Curve.png")),
                      Output(name="log_reg_confusion_matrix", type=sklearn_log_regArtifact_charts.File(name="log_reg_confusion_matrix.png")),
                      Output(name="log_reg_precision_recall", type=sklearn_log_regArtifact_charts.File(name="log_reg_precision_recall.png")),
                      Output(name="log_reg_confusion_matrix", type=sklearn_log_regArtifact_charts.File(name="log_reg_confusion_matrix.png"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name
    )

    # Launch H2O model training
    h2o_results = run_domino_job_task(
        flyte_task_name="Train H2O Model",
        command="flows/h2o_model_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default'])],
        output_specs=[Output(name="model", type=h2oArtifact.File(name="model.pkl")),
                      Output(name="h2o_PI_plot", type=h2oArtifact_charts.File(name="h2o_PI_plot.png")),
                      Output(name="h2o_SHAP_Summary", type=h2oArtifact_charts.File(name="h2o_SHAP_Summary.png"))],
        use_project_defaults_for_omitted=True,
        environment_name=environment_name,
        hardware_tier_name=hardware_tier_name,
    )

    # Launch sklearn random forest training
    sklearn_rf_results = run_domino_job_task(
        flyte_task_name="Train Sklearn RF",
        command="flows/sklearn_RF_train.py",
        inputs=[Input(name='credit_card_default', type=FlyteFile[TypeVar('csv')], value=load_data['credit_card_default']),
                Input(name='num_estimators', type=int, value=100)],
        output_specs=[Output(name="model", type=sklearn_rfArtifact.File(name="model.pkl")),
                      Output(name="rf_ROC_Curve_n_estimators", type=sklearn_rfArtifact_charts.File(name="rf_ROC_Curve_n_estimators.png")),
                      Output(name="rf_confusion_matrix_n_estimators", type=sklearn_rfArtifact_charts.File(name="rf_confusion_matrix_n_estimators.png")),
                      Output(name="rf_precision_recall_n_estimators", type=sklearn_rfArtifact_charts.File(name="rf_precision_recall_n_estimators.png")),
                      Output(name="rf_ROC_Curve", type=sklearn_rfArtifact_charts.File(name="rf_ROC_Curve.png")),
                      Output(name="rf_confusion_matrix", type=sklearn_rfArtifact_charts.File(name="rf_confusion_matrix.png")),
                      Output(name="rf_precision_recall", type=sklearn_rfArtifact_charts.File(name="rf_precision_recall.png"))],
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