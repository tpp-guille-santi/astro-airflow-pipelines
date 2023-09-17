"""DAG that creates the duckdb pool and kicks off the pipeline by producing to the start dataset."""

from datetime import datetime
from datetime import timedelta

from airflow.decorators import dag
from airflow.operators.python import PythonOperator

from include.custom_task_groups.create_bucket import CreateBucket
from include.repositories import BackendRepository
from include.repositories import MinioRepository
from include.repositories import TelegramRepository
from include.settings import settings
from include.train_model.tasks import download_new_images
from include.train_model.tasks import images_over_threshold
from include.train_model.tasks import process_images
from include.train_model.tasks import create_model
from include.train_model.tasks import validate_model
from include.train_model.tasks import transform_model
from include.train_model.tasks import upload_model

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
def train_model():
    backend_repository = BackendRepository(base_url=settings.BACKEND_URL)
    telegram_repository = TelegramRepository(
        base_url=settings.TELEGRAM_URL,
        token=settings.TELEGRAM_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
    )
    minio_repository = MinioRepository(
        host=settings.MINIO_HOST,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
    )
    create_bucket_tg = CreateBucket(
        task_id="create_images_bucket", bucket_name='images'
    )


    images_over_threshold(backend_repository) >> create_bucket_tg >> download_new_images(
        backend_repository, minio_repository) >> process_images() >> create_model(minio_repository           
        ) >> validate_model_task(backend_repository) >> transform_model() >> upload_model()


train_model()
