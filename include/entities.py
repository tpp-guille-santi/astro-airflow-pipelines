import logging

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class Image(BaseModel):
    id: str | None = None
    filename: str | None = None
    material_name: str | None = None
    downloaded: bool | None = None
    tags: list[str] | None = None


class Material(BaseModel):
    name: str
    order: int
    enabled: bool


class ImagesCountResponse(BaseModel):
    count: int


class MLModel(BaseModel):
    timestamp: int
    accuracy: float
