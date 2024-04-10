from flask import Flask, request, Response
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage
from webserver.walls_detection import WallsDetectionService
from webserver.table_extraction import TableExtractionService
from webserver.storage import StorageHandler


app = Flask(__name__)
api = Api(app)


class TaskSchema:
    task_parser = api.parser()
    task_parser.add_argument(
        'image', location='files', type=FileStorage, help='Input image', required=True)
    task_parser.add_argument(
        'type', type=str, help='Task type', required=True, choices=('walls', 'tables'),)


@api.route('/predict')
class TaskController(Resource):
    @api.expect(TaskSchema.task_parser)
    def post(self):
        args = TaskSchema.task_parser.parse_args()
        image_file = args['image']
        type = args['type']

        storage_handler = StorageHandler(image_file)

        if type == 'walls':
            response = WallsDetectionService.predict(
                storage_handler.get_file_dto())

            return Response(response=response, status=200, mimetype='application/json')

        elif type == 'tables':
            response = TableExtractionService.predict(
                storage_handler.get_file_dto())

            return Response(response=response, status=200, mimetype='application/json')

        return Response(status=400)
