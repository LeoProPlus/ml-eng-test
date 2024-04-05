import os
import uuid
from typing import NamedTuple
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'uploads'


class FileStorageDto(NamedTuple):
    id: str
    path: str


class StorageHandler():
    def __init__(self, file) -> None:
        filename = secure_filename(file.filename)
        image_id = str(uuid.uuid4())
        dest_image_dir = os.path.join(UPLOAD_FOLDER, image_id)
        os.makedirs(dest_image_dir)
        dest_path = os.path.join(dest_image_dir, filename)
        file.save(dest_path)
        self.file_dto = FileStorageDto(image_id, dest_path)

    def get_file_dto(self):
        return self.file_dto
