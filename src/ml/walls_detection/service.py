import torch
from torchvision import transforms as T
from PIL import Image
from ml.walls_detection.model import get_pretrained_model
from ml.walls_detection.config import Config


class Service:
    transform = T.ToTensor()
    model = get_pretrained_model()

    def predict(path: str):
        img = Image.open(path)
        img = Service.transform(img)

        with torch.no_grad():
            pred = Service.model([img.to(Config.DEVICE)])

        wall_idxs = torch.where(pred[0]['labels'] == Config.WALL_CLASS_ID)[
            0].cpu().detach().numpy()
        wall_masks = pred[0]['masks'][wall_idxs].cpu().detach().numpy()
        wall_boxes = pred[0]['boxes'][wall_idxs].cpu().detach().numpy()
        wall_scores = pred[0]['scores'][wall_idxs].cpu().detach().numpy()

        return wall_idxs, wall_masks, wall_boxes, wall_scores
