import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes, box_area
from floortrans.loaders.house import House


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True, max_size=9999999999):
        self.is_transform = is_transform
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        self.data_folder = data_folder
        # Load txt file to list
        self.folders = np.genfromtxt(data_folder + data_file, dtype='str')
        self.max_size = max_size

    def __len__(self):
        return min(self.max_size, len(self.folders))

    def __getitem__(self, index):
        sample = self.get_data(index)

        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_data(self, index):
        fplan = cv2.imread(self.data_folder +
                           self.folders[index] + self.image_file_name)
        # correct color channels
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder +
                      self.folders[index] + self.svg_file_name, height, width)
        # Combining them to one numpy tensor
        label_by_class = torch.tensor(house.walls)
        label_by_instance = torch.tensor(house.wall_ids)

        # instances are encoded as different colors
        obj_ids = torch.unique(label_by_instance)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        masks = (label_by_instance == obj_ids[:, None, None]).to(
            dtype=torch.uint8)
        boxes = masks_to_boxes(masks)
        area = box_area(boxes)

        non_empty = torch.where(area > 0)

        non_empty_masks = masks[non_empty]
        non_empty_boxes = boxes[non_empty]

        labels = torch.ones((len(non_empty_boxes),), dtype=torch.int64)
        for i in range(len(non_empty_masks)):
            rows, cols = np.where(non_empty_masks[i])
            labels[i] = label_by_class[rows[0], cols[0]]

        img = torch.tensor(fplan.astype(np.float32))

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            non_empty_boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(non_empty_masks)
        target["labels"] = labels

        return img, target

    def transform(self, sample):
        fplan = sample[0]
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        return (fplan, sample[1])
