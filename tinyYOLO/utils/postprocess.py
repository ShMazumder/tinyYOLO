"""
TinyYOLO Post-Processing
==========================
Decode raw model outputs → bounding boxes + NMS filtering.
"""

import torch
import torchvision

from tinyYOLO.utils.boxcodec import decode_grid


def decode_predictions(outputs, imgsz, conf_thresh=0.25, nc=80, box_mode='ltrb'):
    """
    Decode multi-scale model outputs into bounding boxes.

    Args:
        outputs: List of [B, C, H, W] tensors from model (3 scales).
                 C == nc + 5 -> channels are 4 bbox + 1 obj + nc cls (legacy)
                 C == nc + 4 -> channels are 4 bbox + nc cls (R2, no objectness)
                 The layout is inferred from C and nc, so both heads work here.
        imgsz: Input image size.
        conf_thresh: Confidence threshold for filtering.
        nc: Number of classes.
        box_mode: 'ltrb' or 'exp'. MUST match the head that produced `outputs`;
                  pass the head's `box_mode` attribute.

    Returns:
        List of [N, 6] tensors per batch: (x1, y1, x2, y2, conf, cls).
    """
    batch_size = outputs[0].shape[0]
    all_boxes = [[] for _ in range(batch_size)]

    for scale_idx, pred in enumerate(outputs):
        B, C, H, W = pred.shape

        # Decode boxes: [B, 4, H, W] -> grid-anchored (cx,cy,w,h) in pixels.
        # Uses the SAME codec as the training loss (tinyYOLO.utils.boxcodec) so
        # the parametrization can never diverge between train and inference.
        # The cell index (gi,gj) is added to the center — without it a conv head
        # (translation-equivariant) cannot localize and mAP collapses to ~0.
        pred_box = pred[:, :4, :, :]
        cx, cy, w, h = decode_grid(pred_box, imgsz=imgsz, mode=box_mode)  # each [B, H, W]

        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Infer the channel layout: with objectness C == nc+5, without it C == nc+4.
        has_obj = (C == nc + 5)
        if has_obj:
            pred_obj = torch.sigmoid(pred[:, 4:5, :, :])   # [B, 1, H, W]
            pred_cls = pred[:, 5:, :, :]
        else:
            pred_obj = None
            pred_cls = pred[:, 4:, :, :]

        cls_conf, cls_id = torch.sigmoid(pred_cls).max(dim=1)  # [B, H, W]

        if has_obj:
            # Legacy joint confidence = objectness x class_conf
            cls_conf = cls_conf * pred_obj.squeeze(1)
        # else: the class score IS the confidence — it was trained against a
        # soft IoU-quality target, so it already encodes localisation quality.

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


def postprocess_detections(outputs, conf_thres=0.25, iou_thres=0.45, imgsz=416, box_mode='ltrb'):
    """
    Convenience wrapper to decode predictions, apply NMS, and normalize coordinates to [0, 1].
    """
    # 1. Decode predictions (outputs are in imgsz pixel space)
    # Layout is ambiguous without nc, so assume the R2 head (no objectness).
    # Callers that know nc should call decode_predictions directly.
    nc = outputs[0].shape[1] - 4
    dets_list = decode_predictions(outputs, imgsz=imgsz, conf_thresh=conf_thres, nc=nc,
                                   box_mode=box_mode)
    
    # 2. Apply Non-Maximum Suppression (NMS)
    nms_list = non_max_suppression(dets_list, iou_thresh=iou_thres)
    
    # 3. Normalize coordinates from [0, imgsz] to [0, 1] for relative scaling in plotting
    results = []
    for dets in nms_list:
        if len(dets) > 0:
            dets_norm = dets.clone()
            dets_norm[:, :4] = dets_norm[:, :4] / imgsz
            results.append(dets_norm)
        else:
            results.append(dets)
    return results
