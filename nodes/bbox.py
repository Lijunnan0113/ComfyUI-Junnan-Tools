import inspect

from comfy_api.latest import io

from ._bbox_mask_logic import _build_bbox_masks


def _make_bbox_input():
    """Compatibility shim for BoundingBox.Input across ComfyUI revisions.

    Newer ComfyUI versions accept force_input=True on BoundingBox.Input
    (the upstream RT-DETR DrawBBoxes uses it). Older bundled distributions
    (e.g. ComfyUI-aki-v3) raise TypeError. We probe the constructor
    signature and pass the kwarg only when supported.
    """
    kwargs = {"display_name": "Bounding Boxes"}
    params = inspect.signature(io.BoundingBox.Input.__init__).parameters
    if "force_input" in params:
        kwargs["force_input"] = True
    return io.BoundingBox.Input("bboxes", **kwargs)


class BBoxToMask(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JunnanBBoxToMask",
            display_name="BBox To Mask",
            category="Junnan Tools/Detection",
            description="Convert RT-DETR bounding boxes to per-frame union masks [B, H, W].",
            inputs=[
                _make_bbox_input(),
                io.Image.Input("image", display_name="Reference Image"),
            ],
            outputs=[
                io.Mask.Output("mask"),
            ],
        )

    @classmethod
    def execute(cls, bboxes, image) -> io.NodeOutput:
        import comfy.model_management  # deferred: only at execute time
        batch_size, height, width, _ = image.shape
        masks = _build_bbox_masks(bboxes, batch_size, height, width)
        return io.NodeOutput(masks.to(comfy.model_management.intermediate_device()))
