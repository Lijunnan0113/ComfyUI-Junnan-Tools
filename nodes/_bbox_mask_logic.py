"""Pure bbox→mask conversion. No ComfyUI runtime dependency."""
def _build_bbox_masks(bboxes, batch_size, height, width):
    """Convert bbox list to per-frame union binary masks.

    Args:
        bboxes: accepted shapes:
            - dict: single bbox, single image
            - list[dict]: multiple bboxes, single image
            - list[list[dict]]: per-image bbox lists
            - empty list / None: no detections
        batch_size: target batch dimension (output [B, H, W])
        height, width: spatial dimensions of each mask

    Returns:
        torch.Tensor of shape [batch_size, height, width], float32.
        Each frame is the union of its bboxes (binary 0/1).
        Frames with no detections return all-zero masks.
    """
    import torch  # deferred so the module imports without torch installed
    if isinstance(bboxes, dict):
        bboxes = [[bboxes]]
    elif not isinstance(bboxes, list) or not bboxes:
        bboxes = [[]]
    elif isinstance(bboxes[0], dict):
        bboxes = [bboxes]

    # Compatibility with the original魔改 DrawBBoxes path:
    # broadcast a single-frame bbox list across the image batch.
    if len(bboxes) == 1 and batch_size > 1:
        bboxes = bboxes * batch_size

    # Pad / truncate to maintain image[i] ↔ mask[i] alignment.
    bboxes = (bboxes + [[]] * batch_size)[:batch_size]

    all_masks = []
    for frame in bboxes:
        mask = torch.zeros((height, width), dtype=torch.float32)
        for bbox in frame:
            x1 = max(0, int(bbox["x"]))
            y1 = max(0, int(bbox["y"]))
            x2 = min(width, int(bbox["x"] + bbox["width"]))
            y2 = min(height, int(bbox["y"] + bbox["height"]))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
        all_masks.append(mask)

    return torch.stack(all_masks, dim=0)
