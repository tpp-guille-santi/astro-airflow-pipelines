import logging
from collections import defaultdict

from include.entities import Material
from include.repositories import BackendRepository
from include.repositories import TelegramRepository
from include.settings import settings

LOGGER = logging.getLogger(__name__)


class Usecases:
    def __init__(
        self,
        backend_repository: BackendRepository,
        telegram_repository: TelegramRepository,
    ) -> None:
        self.backend_repository = backend_repository
        self.telegram_repository = telegram_repository

    def get_potential_tags(self) -> dict[str, int]:
        print('get_potential_tags')
        images = self.backend_repository.get_all_images()
        materials = self.backend_repository.get_all_materials()
        materials_names = {material.name for material in materials}
        tags = defaultdict(lambda: 0)
        for image in images:
            if image.tags:
                for tag in image.tags:
                    if tag not in materials_names:
                        tags[tag] += 1
        return tags

    def create_new_materials(self, tags: dict[str, int], threshold: int) -> list[Material]:
        print('create_new_materials')
        new_materials = []
        latest_material = self.backend_repository.get_latest_material()
        latest_order = latest_material.order
        for tag, count in tags.items():
            print(tag, count)
            if count >= threshold:
                print(threshold)
                print('Over threshold')
                latest_order += 1
                material = Material(name=tag, order=latest_order, enabled=False)
                created_material = self.backend_repository.create_material(material)
                new_materials.append(created_material)
        return new_materials

    def send_telegram_notification(self, new_materials: list[Material]):
        print('send_telegram_notification')
        if new_materials:
            formatted_new_materials = '\n'.join(
                ['- ' + material.name for material in new_materials]
            )
            message = f'Se crearon los siguientes materiales:\n{formatted_new_materials}'
            self.telegram_repository.send_message(message)


def task_fail_alert(context):
    task = (context.get('task_instance').task_id,)
    dag = (context.get('task_instance').dag_id,)

    message = f'Falló la Task: <b>{task[0]}</b> del DAG <b>{dag[0]}</b>'
    print(message)
    telegram_repository = TelegramRepository(
        base_url=settings.TELEGRAM_URL,
        token=settings.TELEGRAM_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
    )
    telegram_repository.send_message(message)
