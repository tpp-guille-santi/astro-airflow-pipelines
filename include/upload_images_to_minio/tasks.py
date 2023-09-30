import logging
import os

from airflow.decorators import task

from include.repositories import MinioRepository

LOGGER = logging.getLogger(__name__)


@task.short_circuit
def upload_images_to_minio(directory: str, minio_repository: MinioRepository):
    if not directory:
        return
    # Iterate through the folder
    for material_dir in os.listdir(directory):
        material_path = os.path.join(directory, material_dir)
        print(f'material_path: {material_path}')
        # Check if the item in the main directory is a directory
        if os.path.isdir(material_path):
            # Iterate through the images in the material directory
            for image_file in os.listdir(material_path):
                image_path = os.path.join(material_path, image_file)
                print(f'image_path: {image_path}')
                try:
                    # Generate the object key based on material and image filename
                    object_key = f'{material_dir}/{image_file}'
                    # Upload the image to Minio
                    minio_repository.save_images_from_file(object_key, image_path)
                    print(f'Uploaded: {object_key}')
                except Exception as e:
                    print(f'Error uploading {image_path}: {e}')
