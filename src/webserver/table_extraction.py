import json
from webserver.storage import FileStorageDto
from ml.table_extraction.service import Service


class TableExtractionService:
    def predict(image_storage_dto: FileStorageDto):
        detection_results = Service.predict(image_storage_dto.path)

        resposne = TableExtractionService.build_response(
            detection_results, image_storage_dto)

        return json.dumps(resposne)

    def build_response(detection_results: list, image_storage_dto: FileStorageDto):
        resposne = {
            'type': 'schedule_of_materials',
            'image_id': image_storage_dto.id,
            'detectionResults': detection_results
        }

        return resposne
