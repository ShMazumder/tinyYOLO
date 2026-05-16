"""
TinyYOLO Post-Processing
==========================
Decode raw model outputs → bounding boxes + NMS filtering.
"""

import torch
import torchvision


def decode_predictions(outputs, imgsz, conf_thresh=0.25, nc=80):
    """
    Decode multi-scale model outputs into bounding boxes.

    Args:
        outputs: List of [B, 5+nc, H, W] tensors from model (3 scales).
                 Channels: 4 (bbox) + 1 (obj) + nc (classes)
        imgsz: Input image size.
        conf_thresh: Confidence threshold for filtering.
        nc: Number of classes.

    Returns:
        List of [N, 6] tensors per batch: (x1, y1, x2, y2, conf, cls).
    """
    batch_size = outputs[0].shape[0]
    all_boxes = [[] for _ in range(batch_size)]

    for scale_idx, pred in enumerate(outputs):
        B, C, H, W = pred.shape

        # Decode boxes: [B, 4, H, W]
        # Training loss uses CIoU on sigmoid(pred) vs normalized [0,1] targets,
        # so decode must use sigmoid * imgsz (NOT grid-offset decode)
        pred_box = pred[:, :4, :, :]
        cx = torch.sigmoid(pred_box[:, 0]) * imgsz   # center x in pixels
        cy = torch.sigmoid(pred_box[:, 1]) * imgsz   # center y in pixels
        w = torch.sigmoid(pred_box[:, 2]) * imgsz     # width in pixels
        h = torch.sigmoid(pred_box[:, 3]) * imgsz     # height in pixels

        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Objectness: [B, 1, H, W] — dedicated objectness head at index 4
        pred_obj = torch.sigmoid(pred[:, 4:5, :, :])  # [B, 1, H, W]

        # Class predictions: [B, nc, H, W] — start at index 5 (after 4 bbox + 1 obj)
        pred_cls = pred[:, 5:, :, :]
        cls_conf, cls_id = torch.sigmoid(pred_cls).max(dim=1)  # [B, H, W]

        # Joint confidence = objectness × class_conf
        cls_conf = cls_conf * pred_obj.squeeze(1)  # [B, H, W]

        for b in range(B):
            # Flatten
            boxes = torch.stack([
                x1[b].flatten(), y1[b].flatten(),
                x2[b].flatten(), y2[b].flatten()
            ], dim=1)  # [H*W, 4]
            scores = cls_conf[b].flatten()      # [H*W]
            classes = cls_id[b].flatten().float()  # [H*W]

            # Filter by confidence
            mask = scores > conf_thresh
            if mask.sum() > 0:
                boxes = boxes[mask]
                scores = scores[mask]
                classes = classes[mask]

                # [N, 6]: x1, y1, x2, y2, conf, cls
                dets = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)
                all_boxes[b].append(dets)

    # Concatenate all scales and cap pre-NMS detections
    results = []
    max_pre_nms = 1000  # Safety cap to prevent memory blowup
    for b in range(batch_size):
        if all_boxes[b]:
            dets = torch.cat(all_boxes[b], dim=0)
            # Keep top-k by confidence if too many
            if len(dets) > max_pre_nms:
                topk = dets[:, 4].topk(max_pre_nms).indices
                dets = dets[topk]
            results.append(dets)
        else:
            results.append(torch.zeros(0, 6, device=outputs[0].device))

    return results


def non_max_suppression(detections, iou_thresh=0.45, max_det=300):
    """
    Apply per-class NMS to a list of detections.

    Args:
        detections: List of [N, 6] tensors: (x1, y1, x2, y2, conf, cls).
        iou_thresh: IoU threshold for NMS.
        max_det: Maximum detections to keep.

    Returns:
        List of [M, 6] tensors after NMS.
    """
    results = []
    for dets in detections:
        if len(dets) == 0:
            results.append(dets)
            continue

        boxes = dets[:, :4]
        scores = dets[:, 4]
        classes = dets[:, 5]

        # Per-class NMS using class offset trick
        offset = classes * 4096  # Large offset per class
        boxes_offset = boxes + offset.unsqueeze(1)
        keep = torchvision.ops.nms(boxes_offset, scores, iou_thresh)

        if len(keep) > max_det:
            keep = keep[:max_det]

        results.append(dets[keep])

    return results


def decode_targets(targets, imgsz):
    """
    Decode YOLO-format targets to absolute xyxy format.

    Args:
        targets: [B, max_objects, 5] — (cls, cx, cy, w, h) normalized.
        imgsz: Input image size.

    Returns:
        List of [N, 5] tensors per batch: (x1, y1, x2, y2, cls).
    """
    batch_size = targets.shape[0]
    results = []

    for b in range(batch_size):
        valid = targets[b, :, 2] > 0  # Has width > 0
        t = targets[b][valid]

        if len(t) == 0:
            results.append(torch.zeros(0, 5, device=targets.device))
            continue

        cls = t[:, 0]
        cx = t[:, 1] * imgsz
        cy = t[:, 2] * imgsz
        w = t[:, 3] * imgsz
        h = t[:, 4] * imgsz

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        gt = torch.stack([x1, y1, x2, y2, cls], dim=1)
        results.append(gt)

    return results
