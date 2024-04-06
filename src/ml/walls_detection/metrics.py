import torch
from tqdm import tqdm
from ml.walls_detection.config import Config


def calculate_metrics(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    smooth = 1e-8
    num_samples = 0

    model.eval()
    for i, (imgs, targ) in enumerate(tqdm(loader)):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targ]

        with torch.no_grad():
            pred = model(imgs)

        # iterate over samples in batch
        for j in range(len(pred)):
            # extract predicted wall mask
            predicted_wall_idxs = torch.where(
                pred[j]['labels'] == Config.WALL_CLASS_ID)[0]
            predicted_wall_masks = pred[j]['masks'][predicted_wall_idxs].cpu(
            ).detach()
            predicted_wall_mask = predicted_wall_masks.squeeze().sum(axis=0)
            predicted_wall_mask = predicted_wall_mask > 0
            preds = predicted_wall_mask.short()

            # extract target wall mask
            target_wall_idxs = torch.where(
                targets[j]['labels'] == Config.WALL_CLASS_ID)[0]
            target_wall_masks = targets[j]['masks'][target_wall_idxs].cpu(
            ).detach()
            target_wall_mask = target_wall_masks.sum(axis=0)
            target_wall_mask = target_wall_mask > 0
            y = target_wall_mask.short()

            # calculate metrics
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            num_samples += 1

            intersection = (preds * y).sum()
            total = (preds + y).sum()
            union = total - intersection

            dice_score += (2 * intersection + smooth) / (total + smooth)
            iou_score += (intersection + smooth) / (union + smooth)

    result = {
        "acc": (num_correct/num_pixels*100).item(),
        "dice": (dice_score/num_samples).item(),
        "iou": (iou_score/num_samples).item(),
        "num_samples": num_samples
    }

    return result
