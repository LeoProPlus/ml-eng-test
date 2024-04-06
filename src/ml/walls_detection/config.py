import os
import torch


class Config:
    ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room",
                    "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
    NUM_CLASSES = len(ROOM_CLASSES)
    WALL_CLASS_ID = 2
    MOEDL_FILE = os.getenv('WALLS_DETECTION_MODEL_FILE',
                           'ml/walls_detection/checkpoints/model.pt')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
