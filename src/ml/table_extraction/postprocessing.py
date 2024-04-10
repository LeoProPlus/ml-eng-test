import torch
from torchvision.ops.boxes import nms


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label, iou_threshold=1):
    m = outputs.logits.softmax(-1).max(-1)

    # apply non maximum suppresion
    nms_idxs = nms(boxes=rescale_bboxes(
        outputs['pred_boxes'][0], img_size), scores=m.values[0], iou_threshold=iou_threshold).detach().cpu()

    pred_labels = list(m.indices.detach().cpu().numpy())[0][nms_idxs]
    pred_scores = list(m.values.detach().cpu().numpy())[0][nms_idxs]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0][nms_idxs]
    pred_bboxes = [elem.tolist()
                   for elem in rescale_bboxes(pred_bboxes, img_size)]

    # build objects list
    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects
