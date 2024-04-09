import numpy as np
from tqdm.auto import tqdm


def objects_to_crops(img, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images.
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

        # If table is predicted to be rotated, rotate cropped image:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)

        cropped_table['image'] = cropped_img

        table_crops.append(cropped_table)

    return table_crops


def get_cell_coordinates_by_col(table_data, content_rows, header_rows):
    # Extract rows and columns
    rows = content_rows
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    headers = header_rows

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    headers.sort(key=lambda x: x['bbox'][1])

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


def apply_easyocr_for_cell(cell_image, reader):
    result = reader.readtext(np.array(cell_image))

    if len(result) > 0:
        text = " ".join([x[1] for x in result])
        return text

    return None


def apply_ocr_for_column_cells(cell_coordinates, cropped_table, easyocr_reader):
    # let's OCR col by col
    columns = []

    for idx, row in enumerate(tqdm(cell_coordinates)):
        rows = []
        headers = []

        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))

            # apply OCR
            text = apply_easyocr_for_cell(cell_image, easyocr_reader)

            if text is not None:
                rows.append(text)

        for cell in row["headers"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))

            # apply OCR
            text = apply_easyocr_for_cell(cell_image, easyocr_reader)

            if text is not None:
                headers.append(text)

        columns.append({'header': headers, 'rows': rows})

    return columns


def classify_rows(table_data, epsilon=5):
    headers = [entry for entry in table_data if entry['label']
               == 'table column header']
    rows = [entry for entry in table_data if entry['label'] == 'table row']

    rows.sort(key=lambda x: x['bbox'][1])
    headers.sort(key=lambda x: x['bbox'][1])

    header_rows = []
    content_rows = []
    table_name_row = None
    removed_rows = set()

    if len(headers) > 0 and len(rows) > 0 and headers[0]['bbox'][1] + epsilon > rows[0]['bbox'][1]:
        table_name_row = rows[0]
        removed_rows.add(0)

    for header in headers:
        for i, row in enumerate(rows):
            if header['bbox'][1] - epsilon < rows[i]['bbox'][1] and header['bbox'][3] + epsilon > rows[i]['bbox'][3] and i not in removed_rows:
                header_rows.append(row)
                removed_rows.add(i)

    for i, row in enumerate(rows):
        if i not in removed_rows:
            content_rows.append(row)

    return content_rows, header_rows, table_name_row


def apply_easyocr_for_table_name(cropped_table, table_name_row, easyocr_reader):
    table_name_cell_image = np.array(
        cropped_table.crop(table_name_row["bbox"]))
    table_name = apply_easyocr_for_cell(table_name_cell_image, easyocr_reader)

    return table_name
