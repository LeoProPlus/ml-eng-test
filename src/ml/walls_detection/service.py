import torch
from torchvision import transforms as T
from PIL import Image
from .model import get_model_instance_segmentation
from .checkpoint import laod_model_checkpoint
from .config import Config


class Service:
    transform = T.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model_instance_segmentation(Config.NUM_CLASSES)
    laod_model_checkpoint(Config.MOEDL_FILE, model)
    model.eval()

    def predict(path: str):
        img = Image.open(path)
        img = Service.transform(img)

        with torch.no_grad():
            pred = Service.model([img.to(Service.device)])

        wall_idxs = torch.where(pred[0]['labels'] == Config.WALL_CLASS_ID)[
            0].cpu().detach().numpy()
        wall_masks = pred[0]['masks'][wall_idxs].cpu().detach().numpy()
        wall_boxes = pred[0]['boxes'][wall_idxs].cpu().detach().numpy()
        wall_scores = pred[0]['scores'][wall_idxs].cpu().detach().numpy()

        return wall_idxs, wall_masks, wall_boxes, wall_scores
