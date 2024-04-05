import numpy as np
from tqdm.auto import tqdm


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding,
                bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(
            token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def get_cell_coordinates_by_col(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    headers = [
        entry for entry in table_data if entry['label'] == 'table column header']
    table_projected_row_header = [
        entry for entry in table_data if entry['label'] == 'table projected row header']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    headers.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox']
                     [1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for column in columns:
        column_cells = []
        column_headers = []

        for header in headers:
            cell_bbox = find_cell_coordinates(header, column)
            column_headers.append(
                {'row': header['bbox'], 'cell': cell_bbox, 'is_header': True})

        for row in rows:
            cell_bbox = find_cell_coordinates(row, column)
            column_cells.append(
                {'row': row['bbox'], 'cell': cell_bbox, 'is_header': False})

        # Sort cells in the column by X coordinate
        column_headers.sort(key=lambda x: x['row'][1])
        column_cells.sort(key=lambda x: x['row'][1])

        # Append row information to cell_coordinates
        cell_coordinates.append(
            {'column': column['bbox'], 'cells': column_cells, 'headers': column_headers, 'cell_count': len(column_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['column'][0])

    return cell_coordinates


def apply_ocr_to_cell(cell_image, ocr_processor, ocr_model):
    pixel_values = ocr_processor(
        np.array(cell_image), return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    result = ocr_processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    return result


def apply_ocr(cell_coordinates, cropped_table, ocr_processor, ocr_model):
    # let's OCR row by row
    data = dict()
    max_num_rows = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        col_text = []
        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))

            # apply OCR
            result = apply_ocr_to_cell(cell_image, ocr_processor, ocr_model)

            col_text.append(result)

        if len(col_text) > max_num_rows:
            max_num_rows = len(col_text)

        data[idx] = col_text

    print("Max number of columns:", max_num_rows)

    # pad rows which don't have max_num_rows elements
    # to make sure all rows have the same number of columns
    for row, col_data in data.copy().items():
        if len(col_data) != max_num_rows:
            col_data = col_data + \
                ["" for _ in range(max_num_rows - len(col_data))]
        data[row] = col_data

    return data


def apply_ocr_json(cell_coordinates, cropped_table, ocr_processor, ocr_model):
    # let's OCR row by row
    dict_result = {'type': 'schedule_of_materials',
                   'image_id': 54, 'detectionResults': []}
    table = {'tableName': 'A3', 'columns': []}

    for idx, row in enumerate(tqdm(cell_coordinates)):
        rows = []
        headers = []

        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))

            # apply OCR
            result = apply_ocr_to_cell(cell_image, ocr_processor, ocr_model)

            rows.append(result)

        for cell in row["headers"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))

            # apply OCR
            result = apply_ocr_to_cell(cell_image, ocr_processor, ocr_model)

            headers.append(result)

        table['columns'].append({'header': headers, 'rows': rows})

    dict_result['detectionResults'] = table

    return dict_result
