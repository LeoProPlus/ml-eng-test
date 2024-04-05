import json
from webserver.storage import FileStorageDto
from ml.table_extraction.service import Service


class TableExtractionService:
    def predict(image_dto: FileStorageDto):
        resposne = Service.predict(image_dto.path)

        return json.dumps(resposne)
