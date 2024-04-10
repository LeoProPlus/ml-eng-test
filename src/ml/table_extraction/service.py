import torch
from PIL import Image
import easyocr
import numpy as np
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from .config import Config
from .transform import detection_transform, structure_transform
from .postprocessing import outputs_to_objects
from .utils import objects_to_crops, get_cell_coordinates_by_col, apply_ocr_for_column_cells, classify_rows, apply_easyocr_for_table_name


class Service:
    device = Config.DEVICE

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

    # load ocr reader
    easyocr_reader = easyocr.Reader(['en'])

    def predict(file_path: str):
        image = Image.open(file_path).convert("RGB")

        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(Service.device)

        detection_results = []
        with torch.no_grad():
            # forward pass of table detection
            outputs = Service.detection_model(pixel_values)

            table_objects = outputs_to_objects(
                outputs, image.size, Service.detection_id2label, iou_threshold=1)

            tables_crops = objects_to_crops(
                image, table_objects, Service.detection_class_thresholds, padding=60)

            # for each table
            for i in range(len(tables_crops)):
                cropped_table = tables_crops[i]['image'].convert("RGB")

                pixel_values = structure_transform(cropped_table).unsqueeze(0)
                pixel_values = pixel_values.to(Service.device)

                # forward pass of table structure extraction
                outputs = Service.structure_model(pixel_values)

                structure_objects = outputs_to_objects(
                    outputs, cropped_table.size, Service.structure_id2label, iou_threshold=0.95)

                content_rows, header_rows, table_name_row = classify_rows(
                    structure_objects)
                cell_coordinates = get_cell_coordinates_by_col(
                    structure_objects, content_rows, header_rows)

                # apply ocr for table name
                if table_name_row is not None:
                    table_name = apply_easyocr_for_table_name(
                        cropped_table, table_name_row, Service.easyocr_reader)
                else:
                    table_name = None

                # apply ocr for table columns
                columns = apply_ocr_for_column_cells(
                    cell_coordinates, cropped_table=cropped_table, easyocr_reader=Service.easyocr_reader)

                detection_results.append({
                    'tableName': table_name,
                    'columns': columns
                })

        return detection_results
