import json
import numpy as np
from webserver.storage import FileStorageDto
from ml.walls_detection.service import Service


class WallsDetectionService:

    def predict(image_storage_dto: FileStorageDto):
        wall_idxs, _, wall_boxes, wall_scores = Service.predict(
            image_storage_dto.path)

        resposne = WallsDetectionService.build_response(
            wall_idxs, wall_boxes, wall_scores, image_storage_dto)

        return json.dumps(resposne, cls=NpEncoder)

    def build_response(wall_idxs, wall_boxes, wall_scores, image_storage_dto):
        result = {'type': 'walls', 'image_id': image_storage_dto.id,
                  'detectionResults': {}}

        walls = []
        for i in range(len(wall_idxs)):
            walls.append({
                'wallId': wall_idxs[i],
                'position': {
                    'start': {'x': int(wall_boxes[i][0]), 'y': int(wall_boxes[i][1])},
                    'end': {'x': int(wall_boxes[i][2]), 'y': int(wall_boxes[i][3])}
                },
                'confidence': wall_scores[i],
            })

        result['detectionResults']['walls'] = walls

        return result


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
