# ComfyUI Junnan Tools

Standalone ComfyUI custom-node package providing utility nodes for
detection, image, and MiniMax H3 skill workflows. V3 `ComfyExtension`
registration only.

## Installation

Copy the entire `ComfyUI_Junnan_Tools` directory into your ComfyUI
`custom_nodes/` folder and restart ComfyUI:

```
<ComfyUI>/custom_nodes/ComfyUI_Junnan_Tools/
```

No additional Python dependencies are required.

## Nodes

### BBox To Mask (`JunnanBBoxToMask`)

| Field | Value |
|---|---|
| `node_id` | `JunnanBBoxToMask` |
| `display_name` | `BBox To Mask` |
| `category` | `Junnan Tools/Detection` |
| Inputs | `BOUNDING_BOX` (force), `IMAGE` (force) |
| Output | `MASK` — shape `[B, H, W]`, per-frame union |

Each image's bounding boxes are unioned into one binary mask. Frames
with no detections return an all-zero mask of shape `(H, W)`. Output
batch dimension always matches the input image batch:
`image[i] ↔ mask[i]`.

When given a single-frame bbox list with a multi-image batch, the bbox
list is broadcast to every frame. This preserves the behavior of the
original魔改 `DrawBBoxes` mask-output path.

## Workflow migration

If you previously relied on the魔改 `DrawBBoxes` for mask output:

```
# Before
RT-DETR Detect ──→ DrawBBoxes (魔改) ──→ mask ──→ downstream

# After
RT-DETR Detect ──┬──→ DrawBBoxes (official) ──→ image preview
                 └──→ JunnanBBoxToMask ──→ mask ──→ downstream
```

After migration, restore the official `nodes_rtdetr.py` and delete
your魔改 copy.

### MiniMax H3 Skills (`JunnanMiniMaxH3Skills`)

Reads `SKILL.md` files from a MiniMax H3 `skills` directory and outputs
JSON tool definitions for an LLM Party tool dispatcher. Set `Skills Directory`
to the `skills` directory of the MiniMax H3 repository. The default package
path is `nodes/skills`; it is intentionally empty until you provide skill
files or select another directory.

The node exposes `load_h3_skill` and `read_h3_skill_resource` handlers and
rejects resource paths that leave the selected skill directory.

## Development

```
cd ComfyUI_Junnan_Tools
pytest tests/
```

Pure-logic tests in `tests/test_bbox.py` target
`nodes._bbox_mask_logic._build_bbox_masks` directly — no ComfyUI
runtime stub required.
