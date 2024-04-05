import torch
from PIL import Image
from transformers import AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .transform import detection_transform, structure_transform
from .postprocessing import outputs_to_objects
from .utils import objects_to_crops, get_cell_coordinates_by_col, apply_ocr_json


class Service:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load table detection model
    detection_model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm")
    detection_model.to(device)
    detection_model.eval()

    # update detection_id2label to include "no object"
    detection_id2label = detection_model.config.id2label
    detection_id2label[len(detection_model.config.id2label)] = "no object"

    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }

    # load structure model
    structure_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)
    structure_model.eval()

    # update structure_id2label to include "no object"
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    # load ocr model
    ocr_processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-small-printed")
    ocr_model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-small-printed")
    ocr_model.eval()

    def predict(file_path: str):
        image = Image.open(file_path).convert("RGB")

        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(Service.device)

        with torch.no_grad():
            outputs = Service.detection_model(pixel_values)

            objects = outputs_to_objects(
                outputs, image.size, Service.detection_id2label, iou_threshold=1)

            tokens = []
            tables_crops = objects_to_crops(
                image, tokens, objects, Service.detection_class_thresholds, padding=0)
            cropped_table = tables_crops[0]['image'].convert("RGB")

            pixel_values = structure_transform(cropped_table).unsqueeze(0)
            pixel_values = pixel_values.to(Service.device)

            # forward pass
            outputs = Service.structure_model(pixel_values)

            cells = outputs_to_objects(
                outputs, cropped_table.size, Service.structure_id2label, iou_threshold=0.95)
            cell_coordinates = get_cell_coordinates_by_col(cells)

            result = apply_ocr_json(cell_coordinates, cropped_table=cropped_table,
                                    ocr_processor=Service.ocr_processor, ocr_model=Service.ocr_model)

        return result
