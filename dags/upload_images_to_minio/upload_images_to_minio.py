"""DAG that creates the duckdb pool and kicks off the pipeline by producing to the start dataset."""

from datetime import datetime
from datetime import timedelta

from airflow.decorators import dag

from include.custom_task_groups.create_bucket import CreateBucket
from include.repositories import MinioRepository
from include.settings import settings
from include.upload_images_to_minio.tasks import upload_images_to_minio

default_args = {
    'owner': 'Santiago Gandolfo',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    default_args=default_args,
    tags=['train_model', 'cnn', 'pipeline', 'cnn-pipeline'],
    schedule_interval='0 6 * * *',
    catchup=False,
    max_active_runs=1,
)
def upload_images_to_minio_dag():
    minio_repository = MinioRepository(
        host=settings.MINIO_HOST,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
    )
    create_bucket_tg = CreateBucket(
        task_id="create_images_bucket", bucket_name='images'
    )
    directory = "{{ dag_run.conf.get('directory', '') }}"
    create_bucket_tg >> upload_images_to_minio(directory, minio_repository)


upload_images_to_minio_dag()
