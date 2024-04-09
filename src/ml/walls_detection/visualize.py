from PIL import Image
import matplotlib.pyplot as plt
from ml.walls_detection.service import Service
from matplotlib.patches import Rectangle


def draw_bboxes_on_img(img_path: str):
    img = Image.open(img_path)
    wall_idxs, wall_masks, wall_boxes, wall_scores = Service.predict(img_path)

    plt.imshow(img)
    for box in wall_boxes:
        ax = plt.gca()

        x = box[0]
        y = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = Rectangle((x, y), width, height, linewidth=2,
                         edgecolor='r', facecolor='none')

        ax.add_patch(rect)
