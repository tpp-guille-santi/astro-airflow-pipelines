# --------------- #
# PACKAGE IMPORTS #
# --------------- #

from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from minio import Minio

from include.settings import settings

# -------------------- #
# Local module imports #
# -------------------- #


# --------------- #
# TaskGroup class #
# --------------- #


class CreateBucket(TaskGroup):
    """A task group to create a bucket if it does not already exist."""

    def __init__(self, task_id, task_group=None, bucket_name=None, **kwargs):
        """Instantiate a CreateBucketOperator."""
        super().__init__(group_id=task_id, parent_group=task_group, ui_color='#00A7FB', **kwargs)

        # --------------------- #
        # List Buckets in MinIO #
        # --------------------- #

        @task(task_group=self)
        def list_buckets_minio(minio_client: Minio):
            """Returns the list of all bucket names in a MinIO instance."""
            buckets = minio_client.list_buckets()
            existing_bucket_names = [bucket.name for bucket in buckets]
            print(f'MinIO contains: {existing_bucket_names}')

            return existing_bucket_names

        # -------------------------------------- #
        # Decide if a bucket needs to be created #
        # -------------------------------------- #

        @task.branch(task_group=self)
        def decide_whether_to_create_bucket(buckets):
            """Returns a task_id depending on whether the bucket name provided
            to the class is in the list of buckets provided as an argument."""

            if bucket_name in buckets:
                return f'{task_id}.bucket_already_exists'
            else:
                return f'{task_id}.create_bucket'

        # ------------- #
        # Create Bucket #
        # ------------- #

        @task(task_group=self)
        def create_bucket(minio_client: Minio):
            """Creates a bucket in MinIO."""

            minio_client.make_bucket(bucket_name)

        # ----------------------------- #
        # Empty Operators for structure #
        # ----------------------------- #
        client = Minio(
            settings.MINIO_HOST,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        bucket_already_exists = EmptyOperator(task_id='bucket_already_exists', task_group=self)

        bucket_exists = EmptyOperator(
            task_id='bucket_exists', trigger_rule='none_failed_min_one_success', task_group=self
        )

        # set dependencies within task group
        branch_task = decide_whether_to_create_bucket(list_buckets_minio(client))
        branch_options = [create_bucket(client), bucket_already_exists]
        branch_task >> branch_options >> bucket_exists
