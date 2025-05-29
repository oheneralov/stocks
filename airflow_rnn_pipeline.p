from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="my_rnn_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python /app/scripts/preprocess.py"
    )

    train = BashOperator(
        task_id="train_rnn_model",
        bash_command="python /app/scripts/train.py"
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="python /app/scripts/evaluate.py"
    )

    deploy = BashOperator(
        task_id="deploy_to_ecr",
        bash_command="bash /app/scripts/deploy_to_ecr.sh"
    )

    preprocess >> train >> evaluate >> deploy
