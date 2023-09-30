import io
import json
import logging
import pickle

import firebase_admin
import httpx
import numpy as np
import tensorflow
from firebase_admin import credentials
from firebase_admin import ml
from minio import Minio
from PIL import Image as PilImage
from pillow_heif import register_heif_opener
from tensorflow import keras

from include.entities import Image
from include.entities import ImagesCountResponse
from include.entities import Material
from include.entities import MaterialUpdate
from include.entities import MLModel
from include.settings import settings

register_heif_opener()

LOGGER = logging.getLogger(__name__)


class BackendRepository:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_all_images(self) -> list[Image]:
        with httpx.Client() as client:
            url = f'{self.base_url}/images/'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return [Image(**item) for item in data]
            response.raise_for_status()

    def get_new_images(self) -> list[Image]:
        with httpx.Client() as client:
            url = f'{self.base_url}/images/?downloaded=false'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return [Image(**item) for item in data]
            response.raise_for_status()

    def get_all_materials(self) -> list[Material]:
        with httpx.Client() as client:
            url = f'{self.base_url}/materials/'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return [Material(**item) for item in data]
            response.raise_for_status()

    def get_enabled_materials(self) -> dict[str, Material]:
        with httpx.Client() as client:
            url = f'{self.base_url}/materials/?enabled=true'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return {item['name']: Material(**item) for item in data}
            response.raise_for_status()

    def get_latest_material(self) -> Material:
        with httpx.Client() as client:
            url = f'{self.base_url}/materials/latest/'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return Material(**data)
            response.raise_for_status()

    def create_material(self, material: Material) -> Material:
        with httpx.Client() as client:
            url = f'{self.base_url}/materials/'
            response = client.post(url, json=material.dict())
            if response.status_code == 201:
                data = response.json()
                return Material(**data)
            response.raise_for_status()

    def enable_material(self, material_id: str, material: MaterialUpdate) -> None:
        with httpx.Client() as client:
            url = f'{self.base_url}/materials/{material_id}'
            response = client.patch(url, json=material.dict())
            if response.status_code == 200:
                data = response.json()
                return Material(**data)
            response.raise_for_status()

    def get_new_images_count(self) -> int:
        with httpx.Client() as client:
            url = f'{self.base_url}/images/count/?downloaded=false'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                images_count = ImagesCountResponse(**data)
                return images_count.count
            response.raise_for_status()

    def download_image(self, image: Image) -> io.BytesIO:
        with httpx.Client() as client:
            url = f'{self.base_url}/images/file/{image.filename}/'
            print(url)
            response = client.get(url)
            if response.status_code == 200:
                image_data = response.content
                return io.BytesIO(image_data)
            response.raise_for_status()

    def mark_image_as_downloaded(self, image: Image) -> Image:
        downloaded_image = Image(downloaded=True)
        with httpx.Client() as client:
            response = client.patch(
                f'{self.base_url}/images/{image.id}/', json=downloaded_image.dict()
            )
            if response.status_code == 200:
                data = response.json()
                return Image(**data)
            response.raise_for_status()

    def get_latest_model(self):
        with httpx.Client() as client:
            url = f'{self.base_url}/models/latest/'
            response = client.get(url)
            if response.status_code == 200:
                data = response.json()
                return MLModel(**data)
            response.raise_for_status()

    def create_model(self, model: MLModel):
        with httpx.Client() as client:
            url = f'{self.base_url}/models/'
            response = client.post(url, json=model.dict())
            if response.status_code == 201:
                data = response.json()
                return MLModel(**data)
            response.raise_for_status()


class TelegramRepository:
    def __init__(self, base_url: str, token: str, chat_id: int):
        self.base_url = base_url
        self.token = token
        self.chat_id = chat_id

    def send_message(self, message: str) -> dict:
        with httpx.Client() as client:
            url = f'{self.base_url}{self.token}/sendMessage'
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
            }
            response = client.post(url, data=data)
            if response.status_code == 200:
                data = response.json()
                return data
            response.raise_for_status()


class FirebaseRepository:
    MODEL_ID = '21073965'

    def __init__(self, firebase_credentials: str, firabase_storage_bucket: str):
        cred = credentials.Certificate({**json.loads(firebase_credentials)})
        bucket = {**json.loads(firabase_storage_bucket)}
        firebase_admin.initialize_app(cred, bucket)

    def upload_model(self, model: keras.Model) -> bool:
        existing_model = ml.get_model(model_id=self.MODEL_ID)
        existing_model.model_format = ml.TFLiteFormat(
            model_source=ml.TFLiteGCSModelSource.from_keras_model(model)
        )
        ml.update_model(existing_model)


class MinioRepository:
    def __init__(self, host: str, access_key: str, secret_key: str):
        self.minio_client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

    def save_model(self, model: keras.Model):
        object_key = 'model.keras'
        model_bytes = pickle.dumps(model)
        self.minio_client.put_object(
            bucket_name='models',
            object_name='model.keras',
            data=io.BytesIO(model_bytes),
            length=len(model_bytes),
        )
        print(f'Uploaded {object_key} to MinIO bucket.')

    def download_model(self) -> keras.Model:
        print('Downloading model')
        model_data = self.minio_client.get_object(
            bucket_name='models',
            object_name='model.keras',
        )
        print('Deserializing model')
        model_bytes = model_data.read()
        return pickle.loads(model_bytes)

    def save_images(self, material: Material, image: Image, image_data: io.BytesIO):
        object_key = f'{material.order:02}-{material.name}/{image.filename}'
        self.minio_client.put_object(
            bucket_name='images',
            object_name=object_key,
            data=image_data,
            length=image_data.getbuffer().nbytes,
        )
        print(f'Uploaded {object_key} to MinIO bucket.')

    def save_images_from_file(self, object_key: str, image_path: str):
        self.minio_client.fput_object(
            bucket_name='images', object_name=object_key, file_path=image_path
        )
        print(f'Uploaded {object_key} to MinIO bucket.')

    def prepare_minio_dataset(self, subset):
        images = []
        labels = []
        objects = self.minio_client.list_objects('images', recursive=True)

        # Sort the objects by their names.
        objects = sorted(objects, key=lambda obj: obj.object_name)

        # Create a mapping of subfolder names to labels based on their order.
        label_mapping = {}
        label_counter = 0

        for obj in objects:
            # Split the object's key into parts using '/' as separator.
            subfolder_name = obj.object_name.split('/')[-2]

            # If the subfolder name is not in the label_mapping, add it with a label.
            if subfolder_name not in label_mapping:
                label_mapping[subfolder_name] = label_counter
                label_counter += 1

            # Assign the label based on the label_mapping.
            label = label_mapping[subfolder_name]
            labels.append(label)

            data = self.minio_client.get_object('images', obj.object_name).read()
            img = PilImage.open(io.BytesIO(data))
            img = img.resize((settings.IMG_HEIGHT, settings.IMG_WIDTH))
            img = np.array(img)
            images.append(img)

        images = np.array(images)
        labels = np.array(labels)
        dataset = tensorflow.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(settings.BATCH_SIZE)
        return dataset
