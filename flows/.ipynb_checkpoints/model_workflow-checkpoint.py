from flytekit import workflow
from flytekit.types.file import FlyteFile
from typing import TypeVar, NamedTuple
from flytekitplugins.domino.helpers import run_domino_job_task, Input, Output
from flytekitplugins.domino.task import DatasetSnapshot

# Define outputs for each model training task
training_outputs = NamedTuple(
    "training_outputs",
    sklearn_log_reg=FlyteFile[TypeVar("pkl")],
    h2o_model=FlyteFile[TypeVar("pkl")],
    sklearn_rf=FlyteFile[TypeVar("pkl")],
    xgboost=FlyteFile[TypeVar("pkl")]
)

@workflow
def model_training_workflow() -> training_outputs:
    """
    Workflow that runs multiple model training jobs in parallel.
    Returns trained model files for each algorithm.
    """
    # Launch sklearn logistic regression training
    sklearn_log_reg_results = run_domino_job_task(
        flyte_task_name="Train Sklearn LogReg",
        command="python flows/sklearn_log_reg_train.py",
        output_specs=[Output(name="model", type=FlyteFile[TypeVar("pkl")])],
        use_project_defaults_for_omitted=True,
        dataset_snapshots=[
            DatasetSnapshot(Name="Test_Credit_Default", Id="6781481dbcb42b7e687c98ed", Version=1)
        ]
    )

    # Launch H2O model training
    h2o_results = run_domino_job_task(
        flyte_task_name="Train H2O Model",
        command="python flows/h2o_model_train.py",
        output_specs=[Output(name="model", type=FlyteFile[TypeVar("pkl")])],
        use_project_defaults_for_omitted=True,
        dataset_snapshots=[
            DatasetSnapshot(Name="Test_Credit_Default", Id="6781481dbcb42b7e687c98ed", Version=1)
        ]
    )

    # Launch sklearn random forest training
    sklearn_rf_results = run_domino_job_task(
        flyte_task_name="Train Sklearn RF",
        command="python flows/sklearn_RF_train.py",
        output_specs=[Output(name="model", type=FlyteFile[TypeVar("pkl")])],
        use_project_defaults_for_omitted=True,
        dataset_snapshots=[
            DatasetSnapshot(Name="Test_Credit_Default", Id="6781481dbcb42b7e687c98ed", Version=1)
        ]
    )

    # Launch XGBoost model training
    xgboost_results = run_domino_job_task(
        flyte_task_name="Train XGBoost",
        command="python flows/xgb_model_train.py",
        output_specs=[Output(name="model", type=FlyteFile[TypeVar("pkl")])],
        use_project_defaults_for_omitted=True,
        dataset_snapshots=[
            DatasetSnapshot(Name="Test_Credit_Default", Id="6781481dbcb42b7e687c98ed", Version=1)
        ]
    )

    # Return all model files
    return training_outputs(
        sklearn_log_reg=sklearn_log_reg_results["model"],
        h2o_model=h2o_results["model"],
        sklearn_rf=sklearn_rf_results["model"],
        xgboost=xgboost_results["model"]
    )