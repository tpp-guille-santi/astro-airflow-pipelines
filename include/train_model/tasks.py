import logging
import os
from time import time

import tensorflow as tf
from airflow.decorators import task

from include.entities import Image
from include.entities import MLModel
from include.entities import Material
from include.model import train_and_evaluate_model
from include.repositories import BackendRepository
from include.repositories import FirebaseRepository
from include.repositories import MinioRepository
from include.repositories import TelegramRepository
from include.settings import settings

LOGGER = logging.getLogger(__name__)


@task.short_circuit
def images_over_threshold(backend_repository: BackendRepository):
    images_count = backend_repository.get_new_images_count()
    return images_count > settings.MODEL_RETRAIN_THRESHOLD


@task()
def download_new_images(backend_repository: BackendRepository,
                        minio_repository: MinioRepository,
                        ):
    print('Downloading new Images')
    images = backend_repository.get_new_images()
    materials = backend_repository.get_enabled_materials()
    for image in images:
        material = _get_material(image, materials)
        if material:
            image_data = backend_repository.download_image(image)
            LOGGER.info("Downloaded image. Uploading to MinIO")
            minio_repository.save_images(material=material, image=image, image_data=image_data)
            LOGGER.info(f"Uploaded image {image.filename} to MiniIO")

            # backend_repository.mark_image_as_downloaded(image)


@task()
def process_images():
    print("Dud task")


@task()
def create_model(minio_repository):
    model, accuracy = train_and_evaluate_model(minio_repository)
    if not os.path.isdir(settings.MODELS_PATH):
        os.makedirs(settings.MODELS_PATH)
    model.save(settings.TRAINED_MODEL_PATH)
    return accuracy


@task.short_circuit
def validate_model(backend_repository: BackendRepository, **context):
    value = context["ti"].xcom_pull(key="return_value", task_ids="create_model")
    print("New Accuracy: ", value)
    material = backend_repository.get_latest_model()
    print("Old Accuracy: ", material.accuracy)
    return value > material.accuracy


@task.short_circuit
def upload_model(firebase_repository: FirebaseRepository,
                 backend_repository: BackendRepository,
                 **context,
                 ):
    model_accuracy = context["ti"].xcom_pull(key="return_value", task_ids="create_model")
    model = tf.keras.models.load_model(settings.TRAINED_MODEL_PATH)
    uploaded_model = firebase_repository.upload_model(model)
    if uploaded_model:
        current_timestamp = int(time())
        ml_model = MLModel(timestamp=current_timestamp, accuracy=model_accuracy)
        backend_repository.create_model(ml_model)
        return True
    return False


@task()
def send_telegram_notification(telegram_repository: TelegramRepository, **context):
    model_accuracy = context["ti"].xcom_pull(key="return_value", task_ids="create_model")
    message = f'Se actualizó el modelo. Nueva exactitud: {model_accuracy}'
    telegram_repository.send_message(message)
    print('Sent Telegram Notification')


def _get_material(image: Image, materials: dict[str, Material]):
    if image.material_name in materials:
        return materials[image.material_name]
    for tag in image.tags:
        if tag in materials:
            return materials[tag]
    return None
